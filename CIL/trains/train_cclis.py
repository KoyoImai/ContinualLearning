
import os
import csv
import sys
import math
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from util import AverageMeter
from models.resnet_cifar_co2l import LinearClassifier

logger = logging.getLogger(__name__)



def adjust_learning_rate_cclis(args, optimizer, epoch):
    lr_enc = args.learning_rate
    lr_prot = args.learning_rate_prototypes
    if args.cosine:
        eta_min_enc = lr_enc * (args.lr_decay_rate ** 3)
        eta_min_prot = lr_prot * (args.lr_decay_rate ** 3)
        lr_enc = eta_min_enc + (lr_enc - eta_min_enc) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
        lr_prot = eta_min_prot + (lr_prot - eta_min_prot) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2        
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr_enc = lr_enc * (args.lr_decay_rate ** steps)
            lr_prot = lr_prot * (args.lr_decay_rate ** steps)

    lr_list = [lr_enc, lr_enc, lr_prot]

    for idx, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr_list[idx]


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr_enc = args.warmup_from_enc + p * (args.warmup_to_enc - args.warmup_from_enc)
        lr_prot = args.warmup_from_prot + p * (args.warmup_to_prot - args.warmup_from_prot)
        lr_list = [lr_enc, lr_enc, lr_prot]

        for idx, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_list[idx]







def train_cclis(opt, model, model2, criterion, optimizer, scheduler, train_loader, epoch, subset_sample_num, score_mask, grad_train_loaders, grad_val_loaders,
                gradtask_train_loaders, gradtask_val_loaders, gradreplay_train_loader, gradreplay_val_loader):

    # modelをtrainモードに変更
    model.train()

    losses = AverageMeter()
    distill = AverageMeter()

    distill_type = opt.distill_type

    for idx, (images, labels, importance_weight, index) in enumerate(train_loader):

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # normalize the prototypes
        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task

            w = model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.prototypes.weight.copy_(w)
        

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
        features, output = model(images)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # 現在タスクのクラス
        target_labels = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
        # print("target_labels : ", target_labels)

        # ISSupCon
        loss = criterion(output,
                         features, 
                         labels, 
                         importance_weight, 
                         index, 
                         target_labels=target_labels, 
                         sample_num=subset_sample_num, 
                         score_mask=score_mask,
                         reduction='mean',
                         )

        if distill_type == 'IRD':
            if opt.target_task > 0:
                # IRD (cur)
                labels_mask = labels < min(target_labels)

                features1_prev_task = features[labels_mask] if IRD_type == 'prev' else features

                features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), opt.current_temp)
                logits_mask = torch.scatter(
                    torch.ones_like(features1_sim),
                    1,
                    torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
                    0
                )
                logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
                features1_sim = features1_sim - logits_max1.detach()
                row_size = features1_sim.size(0)
                logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

                # IRD (past)
                with torch.no_grad():
                    features2, _ = model2(images)
                    features2_prev_task = features2[labels_mask] if IRD_type == 'prev' else features2

                    features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), opt.past_temp)
                    logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                    features2_sim = features2_sim - logits_max2.detach()
                    logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

                loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
                loss += opt.distill_power * loss_distill
                distill.update(loss_distill.item(), bsz)
        elif distill_type == 'PRD':
            if opt.target_task > 0:
                all_labels = torch.unique(labels).view(-1, 1)

                prev_all_labels = torch.arange(target_labels[0])
                
                prototypes_mask = torch.scatter(
                    torch.zeros(len(prev_all_labels), opt.n_cls).float(),
                    1,
                    prev_all_labels.view(-1,1),
                    1
                    ).to(device)

                labels_mask = labels < min(target_labels)

                # PRD (cur)
                sim_prev_task = torch.matmul(prototypes_mask, output)

                features1_sim = torch.div(sim_prev_task, opt.current_temp)
                 

                logits_max1, _ = torch.max(features1_sim, dim=0, keepdim=True)
                features1_sim = features1_sim - logits_max1.detach()  # number stability
                row_size = features1_sim.size(0)
                
                logits1 = torch.exp(features1_sim) / torch.exp(features1_sim).sum(dim=0, keepdim=True)

                # PRD (past)
                with torch.no_grad():
                    _, sim2_prev_task = model2(images)
                    sim2_prev_task = torch.matmul(prototypes_mask, sim2_prev_task)

                    features2_sim = torch.div(sim2_prev_task, opt.past_temp)
                    logits_max2, _ = torch.max(features2_sim, dim=0, keepdim=True)
                    features2_sim = features2_sim - logits_max2.detach()
                    logits2 = torch.exp(features2_sim) /  torch.exp(features2_sim).sum(dim=0, keepdim=True)

                loss_distill = (-logits2 * torch.log(logits1)).sum(0).mean()
                loss += opt.distill_power * loss_distill
                distill.update(loss_distill.item(), bsz)
        else:
            raise ValueError("distill type {} is not supported".format(distill_type))

        # update metric
        losses.update(loss.item(), bsz)

        # 現在の学習率
        current_lr = optimizer.param_groups[0]['lr']

        # for i, param_group in enumerate(optimizer.param_groups):
        #         print(f"Param group {i} learning rate: {param_group['lr']}")
        # print("scheduler.get_last_lr(): ", scheduler.get_last_lr())

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # 学習記録の表示
        if (idx+1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'lr {lr:.5f}'.format(
                   epoch, idx + 1, len(train_loader), loss=losses, lr=current_lr))


    # 勾配分析（訓練用）
    if (opt.grad_analysis and epoch == opt.epochs-1) or (opt.grad_analysis and epoch % opt.grad_analysis_freq == 0):
        grad_analysis_is_supcon(opt=opt, model=model, optimizer=optimizer, criterion=criterion, grad_loader=gradreplay_train_loader, epoch=epoch,
                                importance_weight=importance_weight, index=index, subset_sample_num=subset_sample_num, score_mask=score_mask)
        if opt.target_task > 0:
            grad_analysis_distill(opt, model, model2, optimizer, criterion, gradreplay_train_loader, epoch, distill_type)

    return losses.avg, model2







import pandas as pd
from collections import defaultdict

### ===================================================================================================================================================
def write_full_grad_to_csv_by_column_from_sums(grad_sum_dict, grad_count_dict, output_path, epoch):
    rows = []
    for key in grad_sum_dict:
        grad_mean = grad_sum_dict[key] / grad_count_dict[key]
        anchor_label, layer, param_type, index_str = key
        rows.append([anchor_label, layer, param_type, index_str, grad_mean])
    
    df_new = pd.DataFrame(rows, columns=["anchor_label", "layer", "param_type", "index", f"epoch_{epoch}"])

    if os.path.exists(output_path):
        df_old = pd.read_csv(output_path)
        df_merged = pd.merge(df_old, df_new, on=["anchor_label", "layer", "param_type", "index"], how="outer")
    else:
        df_merged = df_new

    df_merged.to_csv(output_path, index=False)


# 新規：行リスト（anchor, layer, param_type, index_str, epoch_val）を直接CSVに列追加でマージ
def write_full_grad_to_csv_from_rows(rows, output_path, epoch):
    df_new = pd.DataFrame(rows, columns=["anchor_label", "layer", "param_type", "index", f"epoch_{epoch}"])
    if os.path.exists(output_path):
        df_old = pd.read_csv(output_path)
        df_merged = pd.merge(df_old, df_new, on=["anchor_label", "layer", "param_type", "index"], how="outer")
    else:
        df_merged = df_new
    df_merged.to_csv(output_path, index=False)


# === 勾配記録対象のフィルタ関数（要素粒度の判定は使わず、パラメータ粒度で使う） ===
def param_should_record(layer_name, param_type):
    if param_type != "weight":
        return False
    if ("bn" in layer_name) or ("shortcut" in layer_name) or ("downsample" in layer_name):
        return False
    return True
# 先頭チャネルを絞りたいとき用（None なら全チャネル）
CHANNEL_LIMIT = None  # 例: 5 にすると先頭5チャネルだけにスライス


def grad_analysis_is_supcon(opt, model, optimizer, criterion, grad_loader, epoch, importance_weight, index, subset_sample_num, score_mask):
    if not (opt.grad_analysis and (epoch == opt.epochs - 1 or epoch % opt.grad_analysis_freq == 0)):
        return

    path = f"{opt.explog_path}/gradreplay/task{opt.target_task}"
    os.makedirs(path, exist_ok=True)

    grad_log_path = f"{path}/grad_epoch{epoch}_issupcon_log.csv"
    full_grad_log_path = f"{path}/grad_epochall_issupcon_full.csv"
    is_new_log_file = not os.path.exists(grad_log_path)

    # ノルム集計（従来形式）
    grad_sum_dict = defaultdict(float)
    grad_count_dict = defaultdict(int)

    # 詳細勾配（高速版）：(label_i, param_idx) → sum(tensor on GPU), count
    detail_sums = {}                 # key=(label_i, i) -> torch.Tensor (GPU)
    detail_counts = defaultdict(int) # key=(label_i, i) -> int

    # === 勾配インデックスマップを事前に構築（requires_grad のみ対象） ===
    grad_dims = []             # 各パラメータの要素数
    param_index_map = {}       # i -> {layer, param_type, shape}
    named_trainable = [(n, p) for (n, p) in model.named_parameters() if p.requires_grad]
    for i, (name, param) in enumerate(named_trainable):

        # print("name: ", name)                 # name:  encoder.conv1.weight
        # print("param[0:5]: ", param[0:5])     #
        # print("param.shape: ", param.shape)   # param.shape:  torch.Size([64, 3, 3, 3])

        layer_name = '.'.join(name.split('.')[:-1])
        param_type = name.split('.')[-1]
        grad_dims.append(param.data.numel())
        param_index_map[i] = {
            "layer": layer_name,
            "param_type": param_type,
            "shape": tuple(param.shape),
        }
    grad_total = sum(grad_dims)

    # 形状確認
    # print("grad_total: ", grad_total)                       # grad_total:  11498432
    # print("len(param_index_map): ", len(param_index_map))   # len(param_index_map):  65
    # print("param_index_map[0]: ", param_index_map[0])       # param_index_map[0]:  {'layer': 'encoder.conv1', 'param_type': 'weight', 'shape': (64, 3, 3, 3)}
    # print("param_index_map[1]: ", param_index_map[1])       # param_index_map[1]:  {'layer': 'encoder.bn1', 'param_type': 'weight', 'shape': (64,)}
    # print("param_index_map[2]: ", param_index_map[2])       # param_index_map[2]:  {'layer': 'encoder.bn1', 'param_type': 'bias', 'shape': (64,)}
    # print("len(grad_dims): ", len(grad_dims))               # len(grad_dims):  65


    # === 学習ループ ===
    # for _, (images, labels, importance_weight, index) in enumerate(grad_loader):
    for _, (images, labels, importance_weight, index) in tqdm(enumerate(grad_loader)):
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        device = images.device

        # prototypes の正規化
        with torch.no_grad():
            w = model.prototypes.weight.data.clone()
            w = torch.nn.functional.normalize(w, dim=1, p=2)
            model.prototypes.weight.copy_(w)

        # forward
        features, output = model(images)

        # 現在タスクのラベル範囲
        target_labels = list(range(opt.target_task * opt.cls_per_task,
                                   (opt.target_task + 1) * opt.cls_per_task))

        # サンプルごとの損失（reduction='grad_analysis'）
        loss = criterion(output, features, labels, importance_weight, index,
                         target_labels=target_labels,
                         sample_num=subset_sample_num,
                         score_mask=score_mask,
                         reduction='grad_analysis')

        # バッチ内を label ごとにまとめる
        label_to_indices = defaultdict(list)
        for i in range(labels.size(0)):
            label_to_indices[labels[i].item()].append(i)

        # === ラベルごとに backward → 勾配ベクトル化(GPU) → 再構築・集計 ===
        for label_i, indices in label_to_indices.items():

            # label_i の損失のみを取り出して総和を計算
            loss_i = loss[indices].sum()

            # 勾配の初期化と損失の逆伝搬
            optimizer.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            loss_i.backward(retain_graph=True)

            # 勾配を1本のベクトルに（GPU上）
            grads = torch.empty(grad_total, dtype=torch.float32, device=device)
            pointer = 0

            # パラメータの勾配を1次元化して grads に格納
            for name, param in named_trainable:
                n_params = param.data.numel()
                if param.grad is not None:
                    grads[pointer:pointer + n_params].copy_(param.grad.detach().view(-1))
                else:
                    grads[pointer:pointer + n_params].zero_()
                pointer += n_params
            # assert pointer == grad_total

            # ベクトルを各パラメータ形状に戻して、GPU上で集計
            pointer = 0
            for i, n_param in enumerate(grad_dims):

                # grad_dims は各パラメータの要素数を格納したリスト
                # print("len(grad_dims): ", len(grad_dims))   # len(grad_dims):  65

                # パラメータのレイヤー名などを取り出す
                meta = param_index_map[i]
                layer_name = meta["layer"]
                param_type = meta["param_type"]
                shape = meta["shape"]
                
                # 確認
                # print("shape: ", shape)

                # meta が示す layer_name，param_type に該当する勾配を取り出し，形状を shape　を元に復元する
                grad_tensor = grads[pointer:pointer + n_param].view(shape)
                pointer += n_param

                # --- ノルム集計（GPUで計算→tolistで一括CPU取り出し） ---
                if grad_tensor.dim() == 4:
                    # out_chごとに |.| を合計
                    abs_sum = grad_tensor.abs().view(grad_tensor.shape[0], -1).sum(dim=1)
                    for j, g in enumerate(abs_sum.tolist()):
                        key = (label_i, layer_name, param_type, str([j]))
                        grad_sum_dict[key] += g
                        grad_count_dict[key] += 1
                elif grad_tensor.dim() == 2:
                    abs_sum = grad_tensor.abs().sum(dim=1)
                    for j, g in enumerate(abs_sum.tolist()):
                        key = (label_i, layer_name, param_type, str([j]))
                        grad_sum_dict[key] += g
                        grad_count_dict[key] += 1
                elif grad_tensor.dim() == 1:
                    for j, g in enumerate(grad_tensor.abs().tolist()):
                        key = (label_i, layer_name, param_type, str([j]))
                        grad_sum_dict[key] += g
                        grad_count_dict[key] += 1

                # --- 詳細勾配：要素ごと辞書更新をやめ、テンソル合算に切替 ---
                if not param_should_record(layer_name, param_type):
                    continue

                # 先頭チャネルだけに絞るならここでスライス（GPU上）
                if CHANNEL_LIMIT is not None and grad_tensor.dim() >= 1 and grad_tensor.shape[0] > CHANNEL_LIMIT:
                    grad_tensor = grad_tensor[:CHANNEL_LIMIT]

                # (アンカークラス，grad_dimsのインデックス)をキーとして使用
                key_li = (label_i, i)
                # print("key_li: ", key_li)   # key_li:  (1, 0)

                # キー key_li が存在しなければ0埋めされたテンソルを作成
                if key_li not in detail_sums:
                    detail_sums[key_li] = torch.zeros_like(grad_tensor, device=device)
                
                # キー key_l に対応したパラメータの勾配を累積する
                detail_sums[key_li] += grad_tensor
                detail_counts[key_li] += 1


    # === ノルムのCSV出力（従来形式） ===
    with open(grad_log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if is_new_log_file:
            writer.writerow(['current task', 'epoch', 'anchor_label', 'layer', 'param_type', 'index', 'grad_sum', 'grad_mean'])
        for key, grad_sum in grad_sum_dict.items():
            grad_mean = grad_sum / len(grad_loader.dataset)
            label_i, layer_name, param_type, index_str = key
            writer.writerow([opt.target_task, epoch, label_i, layer_name, param_type, index_str, grad_sum, grad_mean])

    # === 詳細勾配：GPUで合算したテンソルを最後に一括でCPUへ → 行生成して保存 ===
    rows = []
    for (label_i, i), tensor_sum in detail_sums.items():

        # パラメータのレイヤー名などを取り出す
        meta = param_index_map[i]
        layer_name = meta["layer"]; param_type = meta["param_type"]; shape = meta["shape"]

        # カウント
        cnt = detail_counts[(label_i, i)]
        
        # 平均計算
        mean_tensor = (tensor_sum / cnt).detach().cpu().reshape(-1)

        # 形状確認
        # print("mean_tensor.shape: ", mean_tensor.shape)   # mean_tensor.shape:  torch.Size([1728])

        # flatten index -> multi-index に戻す
        for flat_idx, g in enumerate(mean_tensor.tolist()):
            idx_tuple = np.unravel_index(flat_idx, shape)
            rows.append([label_i, layer_name, param_type, str(list(idx_tuple)), g])

    write_full_grad_to_csv_from_rows(rows, full_grad_log_path, epoch)
### ===================================================================================================================================================



### grads に勾配を平坦化して保存し，形状を戻して記録する方向性（これでも遅い）
# ### ===================================================================================================================================================
# def write_full_grad_to_csv_by_column_from_sums(grad_sum_dict, grad_count_dict, output_path, epoch):
#     rows = []
#     for key in grad_sum_dict:
#         grad_mean = grad_sum_dict[key] / grad_count_dict[key]
#         anchor_label, layer, param_type, index_str = key
#         rows.append([anchor_label, layer, param_type, index_str, grad_mean])
    
#     df_new = pd.DataFrame(rows, columns=["anchor_label", "layer", "param_type", "index", f"epoch_{epoch}"])

#     if os.path.exists(output_path):
#         df_old = pd.read_csv(output_path)
#         df_merged = pd.merge(df_old, df_new, on=["anchor_label", "layer", "param_type", "index"], how="outer")
#     else:
#         df_merged = df_new

#     df_merged.to_csv(output_path, index=False)


# # === 勾配記録対象のフィルタ関数 ===
# def should_record_grad(layer_name, param_type, index_tuple):
#     if param_type != "weight":
#         return False
#     if "bn" in layer_name or "shortcut" in layer_name or "downsample" in layer_name:
#         return False
#     # if index_tuple and index_tuple[0] >= 5:  # 例: 先頭5チャネルのみ
#     #     return False
#     return True


# def grad_analysis_is_supcon(opt, model, optimizer, criterion, grad_loader, epoch, importance_weight, index, subset_sample_num, score_mask):
#     if not (opt.grad_analysis and (epoch == opt.epochs - 1 or epoch % opt.grad_analysis_freq == 0)):
#         return

#     path = f"{opt.explog_path}/gradreplay/task{opt.target_task}"
#     os.makedirs(path, exist_ok=True)

#     grad_log_path = f"{path}/grad_epoch{epoch}_issupcon_log.csv"
#     full_grad_log_path = f"{path}/grad_epochall_issupcon_full.csv"
#     is_new_log_file = not os.path.exists(grad_log_path)

#     # ノルム集計（従来形式）
#     grad_sum_dict = defaultdict(float)
#     grad_count_dict = defaultdict(int)

#     # 詳細記録（要素ごと、平均集計）
#     grad_raw_sum_dict = defaultdict(float)
#     grad_raw_count_dict = defaultdict(int)


#     # === 勾配インデックスマップを事前に構築（requires_grad のみ対象） ===
#     grad_dims = []             # 各パラメータの要素数
#     param_index_map = dict()   # i -> {layer, param_type, shape}
#     for i, (name, param) in enumerate((n_p for n_p in model.named_parameters() if n_p[1].requires_grad)):

#         # print("name: ", name)                 # name:  encoder.conv1.weight
#         # print("param[0:5]: ", param[0:5])     # param[0:5]:  tensor([[[[ 0.0078, -0.0267,  0.0691],
#         # print("param.shape: ", param.shape)   # param.shape:  torch.Size([64, 3, 3, 3])
#         # assert False

#         layer_name = '.'.join(name.split('.')[:-1])
#         param_type = name.split('.')[-1]
#         grad_dims.append(param.data.numel())
#         param_index_map[i] = {
#             "layer": layer_name,
#             "param_type": param_type,
#             "shape": tuple(param.shape),
#         }
#     grad_total = sum(grad_dims)

#     # 形状確認
#     # print("grad_total: ", grad_total)                       # grad_total:  11498432
#     # print("len(param_index_map): ", len(param_index_map))   # len(param_index_map):  65
#     # print("param_index_map[0]: ", param_index_map[0])       # param_index_map[0]:  {'layer': 'encoder.conv1', 'param_type': 'weight', 'shape': (64, 3, 3, 3)}
#     # print("param_index_map[1]: ", param_index_map[1])       # param_index_map[1]:  {'layer': 'encoder.bn1', 'param_type': 'weight', 'shape': (64,)}
#     # print("param_index_map[2]: ", param_index_map[2])       # param_index_map[2]:  {'layer': 'encoder.bn1', 'param_type': 'bias', 'shape': (64,)}

#     for _, (images, labels, importance_weight, index) in tqdm(enumerate(grad_loader)):
        
#         # 画像とラベルをgpuに配置
#         if torch.cuda.is_available():
#             images = images.cuda(non_blocking=True)
#             labels = labels.cuda(non_blocking=True)

#         # prototypes の正規化
#         with torch.no_grad():
#             w = model.prototypes.weight.data.clone()
#             w = torch.nn.functional.normalize(w, dim=1, p=2)
#             model.prototypes.weight.copy_(w)

#         # 特徴量とoutputを出力
#         features, output = model(images)
        
#         # 現在タスクのラベル範囲を獲得
#         target_labels = list(range(opt.target_task * opt.cls_per_task,
#                                    (opt.target_task + 1) * opt.cls_per_task))

#         # 損失を計算
#         loss = criterion(output, features, labels, importance_weight, index,
#                          target_labels=target_labels,
#                          sample_num=subset_sample_num,
#                          score_mask=score_mask,
#                          reduction='grad_analysis')

#         # バッチ内を label ごとにまとめるためのマップ
#         label_to_indices = defaultdict(list)
#         for i in range(labels.size(0)):
#             label_to_indices[labels[i].item()].append(i)

#         # === ラベルごとに backward → 勾配ベクトル化 → 再構築・記録 ===
#         for label_i, indices in label_to_indices.items():

#             # label_iの損失のみを取り出して総和を計算
#             loss_i = loss[indices].sum()

#             # 勾配の初期化と損失のbackward
#             optimizer.zero_grad()
#             model.zero_grad()
#             loss_i.backward(retain_graph=True)

            
#             # --- 安全版：requires_grad なパラメータごとに必ず pointer を進める ---

#             # gradsを初期化（gradsは）
#             grads = torch.empty(grad_total, dtype=torch.float32)
#             pointer = 0
#             for name, param in model.named_parameters():
#                 if not param.requires_grad:
#                     continue
                
                
#                 # print("name: ", name)                 # name:  encoder.conv1.weight
#                 # print("param[0:5]: ", param[0:5])     # param[0:5]:  tensor([[[[ 0.0078, -0.0267,  0.0691],
#                 # print("param.shape: ", param.shape)   # param.shape:  torch.Size([64, 3, 3, 3])
#                 # print("param.grad: ", param.grad)

#                 # paramの要素数を合計して返す
#                 n_params = param.data.numel()
                
#                 # パラメータの勾配を1次元化して grads に格納
#                 if param.grad is not None:
#                     grads[pointer:pointer + n_params].copy_(param.grad.detach().view(-1).cpu())
#                 else:
#                     grads[pointer:pointer + n_params].zero_()
                
#                 # pointer を更新
#                 pointer += n_params
#             # assert pointer == grad_total  # 必要なら検証用

#             # --- ベクトルを各パラメータ形状に戻して、ノルム集計と詳細記録 ---
#             pointer = 0
#             for i, n_param in enumerate(grad_dims):

#                 # print("n_param: ", n_param)     # n_param:  1728

#                 # パラメータのレイヤー名などを取り出す
#                 meta = param_index_map[i]
#                 layer_name = meta["layer"]
#                 param_type = meta["param_type"]
#                 shape = meta["shape"]

#                 # meta が示す layer_name，param_type に該当する勾配を取り出し，形状を shape　を元に復元する
#                 grad_tensor = grads[pointer:pointer + n_param].view(shape)
#                 pointer += n_param
#                 # print("grad_tensor: ", grad_tensor)
               

#                 # ノルム集計（従来の単位に合わせる：out_channel 単位の L1-sum）
#                 if grad_tensor.dim() == 4:
#                     # (out_ch, in_ch, kh, kw) -> out_ch ごとに |.| を合計
#                     abs_sum = grad_tensor.abs().view(grad_tensor.shape[0], -1).sum(dim=1)
#                     for j, g in enumerate(abs_sum):
#                         key = (label_i, layer_name, param_type, str([j]))
#                         grad_sum_dict[key] += float(g.item())
#                         grad_count_dict[key] += 1
#                 elif grad_tensor.dim() == 2:
#                     # (out, in) -> 行方向（out）で合計
#                     abs_sum = grad_tensor.abs().sum(dim=1)
#                     for j, g in enumerate(abs_sum):
#                         key = (label_i, layer_name, param_type, str([j]))
#                         grad_sum_dict[key] += float(g.item())
#                         grad_count_dict[key] += 1
#                 elif grad_tensor.dim() == 1:
#                     # バイアス等：各要素
#                     for j, g in enumerate(grad_tensor.abs()):
#                         key = (label_i, layer_name, param_type, str([j]))
#                         grad_sum_dict[key] += float(g.item())
#                         grad_count_dict[key] += 1

#                 # 詳細記録（要素ごと）— フィルタを通したものだけ平均集計
#                 for index_tuple in np.ndindex(shape):
#                     if not should_record_grad(layer_name, param_type, index_tuple):
#                         continue
#                     grad_value = float(grad_tensor[index_tuple].item())
#                     key = (label_i, layer_name, param_type, str(list(index_tuple)))
#                     grad_raw_sum_dict[key] += grad_value
#                     grad_raw_count_dict[key] += 1

#     # === ノルムのCSV出力（従来形式） ===
#     with open(grad_log_path, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         if is_new_log_file:
#             writer.writerow(['current task', 'epoch', 'anchor_label', 'layer', 'param_type', 'index', 'grad_sum', 'grad_mean'])
#         for key, grad_sum in grad_sum_dict.items():
#             # 元コード準拠：分母は dataset サイズ
#             grad_mean = grad_sum / len(grad_loader.dataset)
#             label_i, layer_name, param_type, index_str = key
#             writer.writerow([opt.target_task, epoch, label_i, layer_name, param_type, index_str, grad_sum, grad_mean])

#     # === 詳細勾配（要素ごと）の列追加形式で出力（平均値） ===
#     write_full_grad_to_csv_by_column_from_sums(grad_raw_sum_dict, grad_raw_count_dict, full_grad_log_path, epoch)
### ===================================================================================================================================================



### ===================================================================================================================================================
# def write_full_grad_to_csv_by_column_from_sums(grad_sum_dict, grad_count_dict, output_path, epoch):
#     rows = []
#     for key in grad_sum_dict:
#         grad_mean = grad_sum_dict[key] / grad_count_dict[key]
#         anchor_label, layer, param_type, index_str = key
#         rows.append([anchor_label, layer, param_type, index_str, grad_mean])
    
#     df_new = pd.DataFrame(rows, columns=["anchor_label", "layer", "param_type", "index", f"epoch_{epoch}"])

#     if os.path.exists(output_path):
#         df_old = pd.read_csv(output_path)
#         df_merged = pd.merge(df_old, df_new, on=["anchor_label", "layer", "param_type", "index"], how="outer")
#     else:
#         df_merged = df_new

#     df_merged.to_csv(output_path, index=False)


# # === 勾配記録対象のフィルタ関数 ===
# def should_record_grad(layer_name, param_type, index_tuple):
#     if param_type != "weight":
#         return False
#     if "bn" in layer_name or "shortcut" in layer_name or "downsample" in layer_name:
#         return False
#     if index_tuple and index_tuple[0] >= 5:  # 例: 先頭5チャネルのみ
#         return False
#     return True

# def grad_analysis_is_supcon(opt, model, optimizer, criterion, grad_loader, epoch, importance_weight, index, subset_sample_num, score_mask):
#     if not (opt.grad_analysis and (epoch == opt.epochs - 1 or epoch % opt.grad_analysis_freq == 0)):
#         return

#     path = f"{opt.explog_path}/gradreplay/task{opt.target_task}"
#     os.makedirs(path, exist_ok=True)

#     grad_log_path = f"{path}/grad_epoch{epoch}_issupcon_log.csv"
#     full_grad_log_path = f"{path}/grad_epochall_issupcon_full.csv"
#     is_new_log_file = not os.path.exists(grad_log_path)

#     grad_sum_dict = defaultdict(float)
#     grad_count_dict = defaultdict(int)

#     grad_raw_sum_dict = defaultdict(float)
#     grad_raw_count_dict = defaultdict(int)

#     print("→ grad_log_path:", grad_log_path)
#     print("→ full_grad_log_path:", full_grad_log_path)

#     for (images, labels, importance_weight, index) in grad_loader:
#         if torch.cuda.is_available():
#             images = images.cuda(non_blocking=True)
#             labels = labels.cuda(non_blocking=True)

#         print("len(index): ", len(index))

#         bsz = labels.shape[0]

#         with torch.no_grad():
#             w = model.prototypes.weight.data.clone()
#             w = nn.functional.normalize(w, dim=1, p=2)
#             model.prototypes.weight.copy_(w)

#         features, output = model(images)
#         target_labels = list(range(opt.target_task * opt.cls_per_task,
#                                    (opt.target_task + 1) * opt.cls_per_task))

#         loss = criterion(output, features, labels, importance_weight, index,
#                          target_labels=target_labels,
#                          sample_num=subset_sample_num,
#                          score_mask=score_mask,
#                          reduction='grad_analysis')

#         label_to_indices = defaultdict(list)
#         for i in range(bsz):
#             label = labels[i].item()
#             label_to_indices[label].append(i)

#         for label_i, indices in label_to_indices.items():
#             loss_i = loss[indices].sum()
#             optimizer.zero_grad()
#             model.zero_grad()
#             loss_i.backward(retain_graph=True)

#             for name, param in model.named_parameters():
#                 if not param.requires_grad or param.grad is None:
#                     continue

#                 param_type = name.split('.')[-1]
#                 layer_name = '.'.join(name.split('.')[:-1])
#                 grad = param.grad.detach().cpu()

#                 # === 集計用（ノルム） ===
#                 if grad.dim() == 4:
#                     grad_reshaped = grad.view(grad.shape[0], -1)
#                     abs_sum = grad_reshaped.abs().sum(dim=1)
#                     for j, g in enumerate(abs_sum):
#                         key = (label_i, layer_name, param_type, str([j]))
#                         grad_sum_dict[key] += g.item()
#                         grad_count_dict[key] += 1
#                 elif grad.dim() == 2:
#                     abs_sum = grad.abs().sum(dim=1)
#                     for j, g in enumerate(abs_sum):
#                         key = (label_i, layer_name, param_type, str([j]))
#                         grad_sum_dict[key] += g.item()
#                         grad_count_dict[key] += 1
#                 elif grad.dim() == 1:
#                     for j, g in enumerate(grad.abs()):
#                         key = (label_i, layer_name, param_type, str([j]))
#                         grad_sum_dict[key] += g.item()
#                         grad_count_dict[key] += 1

#                 # === 詳細記録用（平均集計） ===
#                 print("grad.shape: ", grad.shape)
#                 for index_tuple in np.ndindex(grad.shape):
#                     # print("index_tuple: ", index_tuple)
#                     grad_value = grad[index_tuple].item()
#                     # print("grad_value: ", grad_value)
#                     key = (label_i, layer_name, param_type, str(list(index_tuple)))
#                     grad_raw_sum_dict[key] += grad_value
#                     grad_raw_count_dict[key] += 1
                
#                 # for index_tuple in np.ndindex(grad.shape):
#                 #     if not should_record_grad(layer_name, param_type, index_tuple):
#                 #         continue
#                 #     grad_value = grad[index_tuple].item()
#                 #     key = (label_i, layer_name, param_type, str(list(index_tuple)))
#                 #     grad_raw_sum_dict[key] += grad_value
#                 #     grad_raw_count_dict[key] += 1

#                 # # === 詳細記録（キーを高速化） ===
#                 # prefix = f"{label_i}|{layer_name}|{param_type}"
#                 # for index_tuple in np.ndindex(grad.shape):
#                 #     if not should_record_grad(layer_name, param_type, index_tuple):
#                 #         continue
#                 #     index_str = '|'.join(map(str, index_tuple))
#                 #     key = f"{prefix}|{index_str}"
#                 #     grad_raw_sum_dict[key] += grad[index_tuple].item()
#                 #     grad_raw_count_dict[key] += 1


#     # === ノルムのCSV出力（従来形式） ===
#     with open(grad_log_path, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         if is_new_log_file:
#             writer.writerow(['current task', 'epoch', 'anchor_label', 'layer', 'param_type', 'index', 'grad_sum', 'grad_mean'])

#         for key, grad_sum in grad_sum_dict.items():
#             count = grad_count_dict[key]
#             grad_mean = grad_sum / len(grad_loader.dataset)
#             label_i, layer_name, param_type, index_str = key
#             writer.writerow([opt.target_task, epoch, label_i, layer_name, param_type, index_str, grad_sum, grad_mean])

#     # === 詳細勾配を列追加形式で出力 ===
#     write_full_grad_to_csv_by_column_from_sums(grad_raw_sum_dict, grad_raw_count_dict, full_grad_log_path, epoch)


### ===================================================================================================================================================




## 絶対値の記録は可能（これだけ関数は完成）
# def grad_analysis_is_supcon(opt, model, optimizer, criterion, grad_loader, epoch, importance_weight, index, subset_sample_num, score_mask):
#     if not (opt.grad_analysis and (epoch == opt.epochs - 1 or epoch % opt.grad_analysis_freq == 0)):
#         return

#     path = f"{opt.explog_path}/gradreplay/task{opt.target_task}"
#     os.makedirs(path, exist_ok=True)
#     grad_log_path = f"{path}/grad_epoch{epoch}_issupcon_log.csv"
#     is_new_file = not os.path.exists(grad_log_path)
#     print("grad_log_path: ", grad_log_path)

#     grad_sum_dict = defaultdict(float)
#     grad_count_dict = defaultdict(int)

#     for (images, labels, importance_weight, index) in grad_loader:
        
#         if torch.cuda.is_available():
#             images = images.cuda(non_blocking=True)
#             labels = labels.cuda(non_blocking=True)
#         bsz = labels.shape[0]

#         # normalize the prototypes
#         with torch.no_grad():
#             prev_task_mask = labels < opt.target_task * opt.cls_per_task

#             w = model.prototypes.weight.data.clone()
#             w = nn.functional.normalize(w, dim=1, p=2)
#             model.prototypes.weight.copy_(w)

#         features, output = model(images)

#         # 現在タスクのクラス
#         target_labels = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))

#         # ISSupCon
#         loss = criterion(output,
#                         features, 
#                         labels, 
#                         importance_weight, 
#                         index, 
#                         target_labels=target_labels, 
#                         sample_num=subset_sample_num, 
#                         score_mask=score_mask,
#                         reduction='grad_analysis',
#                         )
#         # print("loss.shape: ", loss.shape)  # loss.shape:  torch.Size([500]) <-- バッチサイズ

#         # ラベルごとに index をまとめる
#         label_to_indices = defaultdict(list)
#         for i in range(bsz):
#             label = labels[i].item()
#             label_to_indices[label].append(i)



#         for label_i, indices in label_to_indices.items():
#             # 指定ラベルの損失を平均して backward
#             # loss_i = loss_tensor[:, indices].mean()
#             loss_i = loss[indices].sum()
#             optimizer.zero_grad()
#             model.zero_grad()
#             loss_i.backward(retain_graph=True)

#             for name, param in model.named_parameters():
#                 if not param.requires_grad:
#                     continue

#                 param_type = name.split('.')[-1]
#                 layer_name = '.'.join(name.split('.')[:-1])
#                 grad = param.grad.detach().cpu()

#                 if grad.dim() == 4:
#                     grad = grad.view(grad.shape[0], -1)
#                     abs_sum = grad.abs().sum(dim=1)
#                     for j, g in enumerate(abs_sum):
#                         key = (label_i, layer_name, param_type, str([j]))
#                         grad_sum_dict[key] += g.item()
#                         grad_count_dict[key] += 1

#                 elif grad.dim() == 2:
#                     abs_sum = grad.abs().sum(dim=1)
#                     for j, g in enumerate(abs_sum):
#                         key = (label_i, layer_name, param_type, str([j]))
#                         grad_sum_dict[key] += g.item()
#                         grad_count_dict[key] += 1

#                 elif grad.dim() == 1:
#                     for j, g in enumerate(grad.abs()):
#                         key = (label_i, layer_name, param_type, str([j]))
#                         grad_sum_dict[key] += g.item()
#                         grad_count_dict[key] += 1
                


#     #  最終的に平均値をCSVに書き出し
#     with open(grad_log_path, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         if is_new_file:
#             writer.writerow(['current task', 'epoch', 'anchor_label', 'layer', 'param_type', 'index', 'grad_sum', 'grad_mean'])

#         for key, grad_sum in grad_sum_dict.items():
#             count = grad_count_dict[key]
#             # grad_mean = grad_sum / count if count > 0 else 0.0
#             grad_mean = grad_sum / len(grad_loader.dataset)
#             label_i, layer_name, param_type, index_str = key
#             writer.writerow([
#                 opt.target_task,
#                 epoch,
#                 label_i,
#                 layer_name,
#                 param_type,
#                 index_str,
#                 grad_sum,
#                 grad_mean
#             ])


### ===================================================================================================================================================


def grad_analysis_distill(opt, model, model2, optimizer, criterion, grad_loader, epoch, distill_type):
    if not (opt.grad_analysis and (epoch == opt.epochs - 1 or epoch % opt.grad_analysis_freq == 0)):
        return

    path = f"{opt.explog_path}/gradreplay/task{opt.target_task}"
    os.makedirs(path, exist_ok=True)
    grad_log_path = f"{path}/grad_epoch{epoch}_distill_log.csv"
    is_new_file = not os.path.exists(grad_log_path)
    print("grad_log_path: ", grad_log_path)

    grad_sum_dict = defaultdict(float)
    grad_count_dict = defaultdict(int)

    for (images, labels, _, _) in grad_loader:

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.no_grad():
            # normalize the prototypes
            w = model.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.prototypes.weight.copy_(w)

        features, output = model(images)
        device = output.device

        if opt.target_task == 0:
            continue

        target_labels = list(range(opt.target_task * opt.cls_per_task, (opt.target_task + 1) * opt.cls_per_task))
        prev_all_labels = torch.arange(target_labels[0])

        # プロトタイプマスクを構築
        # prototypes_mask = torch.zeros(len(prev_all_labels), opt.n_cls, device=device)
        # prototypes_mask.scatter_(1, prev_all_labels.view(-1, 1), 1)
        prototypes_mask = torch.scatter(
                    torch.zeros(len(prev_all_labels), opt.n_cls).float(),
                    1,
                    prev_all_labels.view(-1,1),
                    1
                    ).to(device)

        # PRD (cur)
        # sim_prev_task = torch.matmul(prototypes_mask, output)  # [prev_cls, batch]
        # features1_sim = sim_prev_task / opt.current_temp
        # logits_max1, _ = torch.max(features1_sim, dim=0, keepdim=True)
        # features1_sim = features1_sim - logits_max1.detach()
        # logits1 = torch.exp(features1_sim) / torch.exp(features1_sim).sum(dim=0, keepdim=True)  # shape: [prev_cls, batch]
        sim_prev_task = torch.matmul(prototypes_mask, output)
        features1_sim = torch.div(sim_prev_task, opt.current_temp)
        logits_max1, _ = torch.max(features1_sim, dim=0, keepdim=True)
        features1_sim = features1_sim - logits_max1.detach()  # number stability
        row_size = features1_sim.size(0)
        logits1 = torch.exp(features1_sim) / torch.exp(features1_sim).sum(dim=0, keepdim=True)

        # PRD (past)
        with torch.no_grad():
            # _, sim2_prev_task = model2(images)
            # sim2_prev_task = torch.matmul(prototypes_mask, sim2_prev_task)
            # features2_sim = sim2_prev_task / opt.past_temp
            # logits_max2, _ = torch.max(features2_sim, dim=0, keepdim=True)
            # features2_sim = features2_sim - logits_max2.detach()
            # logits2 = torch.exp(features2_sim) / torch.exp(features2_sim).sum(dim=0, keepdim=True)
            _, sim2_prev_task = model2(images)
            sim2_prev_task = torch.matmul(prototypes_mask, sim2_prev_task)
            features2_sim = torch.div(sim2_prev_task, opt.past_temp)
            logits_max2, _ = torch.max(features2_sim, dim=0, keepdim=True)
            features2_sim = features2_sim - logits_max2.detach()
            logits2 = torch.exp(features2_sim) /  torch.exp(features2_sim).sum(dim=0, keepdim=True)

        # サンプルごとの PRD loss（バッチ次元）
        loss_vec = (-logits2 * torch.log(logits1)).sum(0)  # shape: [batch_size]
        # print("loss_vec.shape: ", loss_vec.shape)          # loss_vec.shape:  torch.Size([500])

        # ラベルごとに index をまとめる
        label_to_indices = defaultdict(list)
        for i in range(bsz):
            label = labels[i].item()
            label_to_indices[label].append(i)

        # 各ラベルに対して勾配を計算・記録
        for label_i, indices in label_to_indices.items():
            loss_i = loss_vec[indices].sum()
            optimizer.zero_grad()
            model.zero_grad()
            loss_i.backward(retain_graph=True)

            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue

                param_type = name.split('.')[-1]
                layer_name = '.'.join(name.split('.')[:-1])
                grad = param.grad.detach().cpu()

                if grad.dim() == 4:
                    grad = grad.view(grad.shape[0], -1)
                    abs_sum = grad.abs().sum(dim=1)
                    for j, g in enumerate(abs_sum):
                        key = (label_i, layer_name, param_type, str([j]))
                        grad_sum_dict[key] += g.item()
                        grad_count_dict[key] += 1

                elif grad.dim() == 2:
                    abs_sum = grad.abs().sum(dim=1)
                    for j, g in enumerate(abs_sum):
                        key = (label_i, layer_name, param_type, str([j]))
                        grad_sum_dict[key] += g.item()
                        grad_count_dict[key] += 1

                elif grad.dim() == 1:
                    for j, g in enumerate(grad.abs()):
                        key = (label_i, layer_name, param_type, str([j]))
                        grad_sum_dict[key] += g.item()
                        grad_count_dict[key] += 1

    # 平均値をCSVに出力
    with open(grad_log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(['current task', 'epoch', 'anchor_label', 'layer', 'param_type', 'index', 'grad_sum', 'grad_mean'])

        for key, grad_sum in grad_sum_dict.items():
            count = grad_count_dict[key]
            grad_mean = grad_sum / len(grad_loader.dataset)
            label_i, layer_name, param_type, index_str = key
            writer.writerow([
                opt.target_task,
                epoch,
                label_i,
                layer_name,
                param_type,
                index_str,
                grad_sum,
                grad_mean
            ])








# def grad_analysis_supcon(opt, model, optimizer, criterion, grad_loaders, epoch, importance_weight, index, subset_sample_num, score_mask):

#     if (opt.grad_analysis and epoch == opt.epochs-1) or (opt.grad_analysis and epoch % opt.grad_analysis_freq == 0):

#         grad_log_path = f"{opt.explog_path}/gradtask_train_supcon_log.csv"
#         is_new_file = not os.path.exists(grad_log_path)
#         print("grad_log_path: ", grad_log_path)

#         with open(grad_log_path, mode='a', newline='') as f:

#             writer = csv.writer(f)
#             if is_new_file:
#                 writer.writerow(['epoch', 'task', 'layer', 'param_type', 'index', 'grad_value'])

#             for taskid, loader in enumerate(grad_loaders):
                
#                 # 勾配を初期化
#                 optimizer.zero_grad()
#                 model.zero_grad()
                
#                 for (images, labels, importance_weight, index) in loader:

#                     if torch.cuda.is_available():
#                         images = images.cuda(non_blocking=True)
#                         labels = labels.cuda(non_blocking=True)
                    
#                     bsz = labels.shape[0]

#                     # normalize the prototypes
#                     with torch.no_grad():
#                         prev_task_mask = labels < opt.target_task * opt.cls_per_task

#                         w = model.prototypes.weight.data.clone()
#                         w = nn.functional.normalize(w, dim=1, p=2)
#                         model.prototypes.weight.copy_(w)
                    
#                     features, output = model(images)

#                     device = (torch.device('cuda')
#                             if features.is_cuda
#                             else torch.device('cpu'))
                    

#                     # 現在タスクのクラス
#                     target_labels = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))

#                     # ISSupCon
#                     loss = criterion(output,
#                                     features, 
#                                     labels, 
#                                     importance_weight, 
#                                     index, 
#                                     target_labels=target_labels, 
#                                     sample_num=subset_sample_num, 
#                                     score_mask=score_mask)
                    
                    
#                     loss.backward()

#                 # 勾配情報をカーネル単位で出力
#                 for name, param in model.named_parameters():
#                     if param.requires_grad:

#                         param_type = name.split('.')[-1]  # パラメータのタイプ（例: weight, bias）
#                         layer_name = '.'.join(name.split('.')[:-1])  # レイヤー名
#                         grad = param.grad.detach().cpu()
                

#                         if grad.dim() == 4:  # Conv: [out_ch, in_ch, kH, kW]
#                             grad = grad.view(grad.shape[0], -1)  # [out_ch, *]
#                             abs_sum = grad.abs().sum(dim=1)
#                             for i, g in enumerate(abs_sum):
#                                 writer.writerow([
#                                     epoch,
#                                     int(taskid),  # タスクID
#                                     layer_name,
#                                     param_type,
#                                     str([i]),  # カーネル index
#                                     g.item()
#                                 ])

#                         elif grad.dim() == 2:  # Linear: [out_dim, in_dim]
#                             abs_sum = grad.abs().sum(dim=1)
#                             for i, g in enumerate(abs_sum):
#                                 writer.writerow([
#                                     epoch,
#                                     int(taskid),  # タスクID
#                                     layer_name,
#                                     param_type,
#                                     str([i]),  # 出力ユニット index
#                                     g.item()
#                                 ])

#                         elif grad.dim() == 1:  # Bias: [N]
#                             for i, g in enumerate(grad.abs()):
#                                 writer.writerow([
#                                     epoch,
#                                     int(taskid),  # タスクID
#                                     layer_name,
#                                     param_type,
#                                     str([i]),
#                                     g.item()
#                                 ])






# def grad_analysis_distill(opt, model, model2, optimizer, criterion, grad_loaders, epoch, distill_type):

#     if (opt.grad_analysis and epoch == opt.epochs-1) or (opt.grad_analysis and epoch % opt.grad_analysis_freq == 0):

#         grad_log_path = f"{opt.explog_path}/gradtask_train_distill_log.csv"
#         is_new_file = not os.path.exists(grad_log_path)
#         print("grad_log_path: ", grad_log_path)

#         with open(grad_log_path, mode='a', newline='') as f:

#             writer = csv.writer(f)
#             if is_new_file:
#                 writer.writerow(['current task', 'epoch', 'task', 'layer', 'param_type', 'index', 'grad_value'])

#             for taskid, loader in enumerate(grad_loaders):

#                 # 勾配を初期化
#                 optimizer.zero_grad()
#                 model.zero_grad()

#                 for (images, labels, importance_weight, index) in loader:

#                     if torch.cuda.is_available():
#                         images = images.cuda(non_blocking=True)
#                         labels = labels.cuda(non_blocking=True)
                    
#                     bsz = labels.shape[0]

#                     # normalize the prototypes
#                     with torch.no_grad():
#                         prev_task_mask = labels < opt.target_task * opt.cls_per_task

#                         w = model.prototypes.weight.data.clone()
#                         w = nn.functional.normalize(w, dim=1, p=2)
#                         model.prototypes.weight.copy_(w)
                    
#                     features, output = model(images)

#                     device = (torch.device('cuda')
#                             if features.is_cuda
#                             else torch.device('cpu'))
                    
#                     # 現在タスクのクラス
#                     target_labels = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
#                     print("target_labels : ", target_labels)


#                     if distill_type == 'IRD':
#                         if opt.target_task > 0:
#                             # IRD (cur)
#                             labels_mask = labels < min(target_labels)

#                             features1_prev_task = features[labels_mask] if IRD_type == 'prev' else features

#                             features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), opt.current_temp)
#                             logits_mask = torch.scatter(
#                                 torch.ones_like(features1_sim),
#                                 1,
#                                 torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
#                                 0
#                             )
#                             logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
#                             features1_sim = features1_sim - logits_max1.detach()
#                             row_size = features1_sim.size(0)
#                             logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

#                             # IRD (past)
#                             with torch.no_grad():
#                                 features2, _ = model2(images)
#                                 features2_prev_task = features2[labels_mask] if IRD_type == 'prev' else features2

#                                 features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), opt.past_temp)
#                                 logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
#                                 features2_sim = features2_sim - logits_max2.detach()
#                                 logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

#                             loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
#                             loss = opt.distill_power * loss_distill
                    
#                     # プロトタイプ蒸留損失
#                     elif distill_type == 'PRD':
#                         if opt.target_task > 0:

#                             # 全ての種類のラベルを獲得
#                             all_labels = torch.unique(labels).view(-1, 1)

#                             # 過去タスクのすべてのクラス
#                             prev_all_labels = torch.arange(target_labels[0])
                            
#                             # プロトタイプ重みに対して，過去クラスだけを抽出するマスクを作成
#                             prototypes_mask = torch.scatter(
#                                 torch.zeros(len(prev_all_labels), opt.n_cls).float(),
#                                 1,
#                                 prev_all_labels.view(-1,1),
#                                 1
#                                 ).to(device)
#                             # print("prototypes_mask.shape: ", prototypes_mask.shape)    # prototypes_mask.shape:  torch.Size([2, 10])
#                             # print("prototypes_mask: ", prototypes_mask)
#                             # prototypes_mask:  tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                             #                             [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')

#                             labels_mask = labels < min(target_labels)

#                             # PRD (cur)
#                             sim_prev_task = torch.matmul(prototypes_mask, output)
#                             # print("output.shape: ", output.shape)                # output.shape:  torch.Size([10, 1000])
#                             # print("sim_prev_task.shape: ", sim_prev_task.shape)  # sim_prev_task.shape:  torch.Size([2, 1000])

#                             features1_sim = torch.div(sim_prev_task, opt.current_temp)
#                             # print("features1_sim.shape: ", features1_sim.shape)         # features1_sim.shape:  torch.Size([2, 1000])
                            

#                             # 数値的安定性
#                             logits_max1, _ = torch.max(features1_sim, dim=0, keepdim=True)
#                             features1_sim = features1_sim - logits_max1.detach()  # number stability
#                             row_size = features1_sim.size(0)
                            
#                             logits1 = torch.exp(features1_sim) / torch.exp(features1_sim).sum(dim=0, keepdim=True)
#                             # print("logits1.shape: ", logits1.shape)     # logits1.shape:  torch.Size([2, 1000])

#                             # PRD (past)
#                             with torch.no_grad():
#                                 _, sim2_prev_task = model2(images)
#                                 sim2_prev_task = torch.matmul(prototypes_mask, sim2_prev_task)

#                                 features2_sim = torch.div(sim2_prev_task, opt.past_temp)
#                                 logits_max2, _ = torch.max(features2_sim, dim=0, keepdim=True)
#                                 features2_sim = features2_sim - logits_max2.detach()
#                                 logits2 = torch.exp(features2_sim) /  torch.exp(features2_sim).sum(dim=0, keepdim=True)

#                             loss_distill = (-logits2 * torch.log(logits1)).sum(0).mean()
#                             loss = opt.distill_power * loss_distill

#                     else:
#                         raise ValueError("distill type {} is not supported".format(distill_type))
                    
#                     loss.backward()

#                 # 勾配情報をカーネル単位で出力
#                 for name, param in model.named_parameters():
#                     if param.requires_grad:

#                         param_type = name.split('.')[-1]  # パラメータのタイプ（例: weight, bias）
#                         layer_name = '.'.join(name.split('.')[:-1])  # レイヤー名
#                         grad = param.grad.detach().cpu()
                

#                         if grad.dim() == 4:  # Conv: [out_ch, in_ch, kH, kW]
#                             grad = grad.view(grad.shape[0], -1)  # [out_ch, *]
#                             abs_sum = grad.abs().sum(dim=1)
#                             for i, g in enumerate(abs_sum):
#                                 writer.writerow([
#                                     opt.target_task,  # 現在のタスク
#                                     epoch,
#                                     int(taskid),  # タスクID
#                                     layer_name,
#                                     param_type,
#                                     str([i]),  # カーネル index
#                                     g.item()
#                                 ])

#                         elif grad.dim() == 2:  # Linear: [out_dim, in_dim]
#                             abs_sum = grad.abs().sum(dim=1)
#                             for i, g in enumerate(abs_sum):
#                                 writer.writerow([
#                                     opt.target_task,  # 現在のタスク
#                                     epoch,
#                                     int(taskid),  # タスクID
#                                     layer_name,
#                                     param_type,
#                                     str([i]),  # 出力ユニット index
#                                     g.item()
#                                 ])

#                         elif grad.dim() == 1:  # Bias: [N]
#                             for i, g in enumerate(grad.abs()):
#                                 writer.writerow([
#                                     opt.target_task,  # 現在のタスク
#                                     epoch,
#                                     int(taskid),  # タスクID
#                                     layer_name,
#                                     param_type,
#                                     str([i]),
#                                     g.item()
#                                 ])



def val_cclis(opt, model, model2, linear_loader, val_loader, taskil_loaders, epoch):

    # classifierの準備
    classifier = LinearClassifier(name="resnet18", num_classes=opt.n_cls, seed=opt.seed)
    if torch.cuda.is_available():
        classifier = classifier.cuda()
    
    # classifierのOptimizer
    optimizer = optim.SGD(classifier.parameters(),
                          lr=opt.linear_lr,
                          momentum=opt.linear_momentum,
                          weight_decay=opt.linear_weight_decay)

    # schedulerの設定
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 75, 90], gamma=0.2)

    # 損失関数の作成
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, opt.linear_epochs):

        # modelをevalモード，classifierをtrainモードに変更
        model.eval()
        classifier.train()

        losses = AverageMeter()

        # 1エポック分の学習
        for idx, (images, labels) in enumerate(linear_loader):

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # 特徴量獲得
            with torch.no_grad():
                features = model.encoder(images)
            output = classifier(features.detach())
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            # cnt += bsz

            # 最適化ステップ
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 現在の学習率
            current_lr = optimizer.param_groups[0]['lr']

            # 学習記録の表示
            if (idx+1) % opt.print_freq == 0 or idx+1 == len(linear_loader):
                print('Train: [{0}][{1}/{2}]\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                      epoch, idx + 1, len(linear_loader), loss=losses))


        # 検証（これまでの全てのタスクを使用）
        model.eval()
        classifier.eval()

        losses = AverageMeter()

        corr = [0.] * (opt.target_task + 1) * opt.cls_per_task
        cnt  = [0.] * (opt.target_task + 1) * opt.cls_per_task
        correct_task = 0.0

        with torch.no_grad():
            for idx, (images, labels) in enumerate(val_loader):
                images = images.float().cuda()
                labels = labels.cuda()
                bsz = labels.shape[0]

                # forward
                output = classifier(model.encoder(images))
                loss = criterion(output, labels)

                # update metric
                losses.update(loss.item(), bsz)

                #
                cls_list = np.unique(labels.cpu())
                correct_all = (output.argmax(1) == labels)

                for tc in cls_list:
                    mask = labels == tc
                    correct_task += (output[mask, (tc // opt.cls_per_task) * opt.cls_per_task : ((tc // opt.cls_per_task)+1) * opt.cls_per_task].argmax(1) == (tc % opt.cls_per_task)).float().sum()

                for c in cls_list:
                    mask = labels == c
                    corr[c] += correct_all[mask].float().sum().item()
                    cnt[c] += mask.float().sum().item()
                
                # if idx % opt.print_freq == 0:
                #     print('Test: [{0}/{1}]\t'
                #         'Acc@1 {top1:.3f} {task_il:.3f}\t'
                #         'lr {lr:.5f}'.format(
                #             idx, len(val_loader),top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100., lr=current_lr
                #         ))
                print('Test: [{0}/{1}]\t'
                    'Acc@1 {top1:.3f} {task_il:.3f}\t'
                    'lr {lr:.5f}'.format(
                        idx, len(val_loader),top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100., lr=current_lr
                    ))
        print(' * Acc@1 {top1:.3f} {task_il:.3f}'.format(top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.))

        # 学習率の調整
        scheduler.step()

    # 検証（これまで学習した各タスク毎に）
    all_task_accuracies, all_task_losses = taskil_val_cclis(opt, model, classifier, criterion, taskil_loaders)

    classil_acc = np.sum(corr)/np.sum(cnt)*100.
    taskil_acc = correct_task/np.sum(cnt)*100.
    return classil_acc, taskil_acc, all_task_accuracies, all_task_losses


def taskil_val_cclis(opt, model, classifier,  criterion, val_loaders):

    # modelをevalモードに変更
    model.eval()

    all_task_accuracies = []
    all_task_losses = []

    for taskid, val_loader in enumerate(val_loaders):

        losses = AverageMeter()
        correct = 0
        total = 0
        task_accuracy = 0

        with torch.no_grad():

            for idx, (images, labels) in enumerate(val_loader):

                images = images.float().cuda()
                labels = labels.cuda()
                bsz = labels.shape[0]

                y_pred = classifier(model.encoder(images))

                # 出力のクラス範囲を制限
                start_class = idx * opt.cls_per_task
                end_class = (idx+1) * opt.cls_per_task
                y_task = y_pred[:, start_class:end_class]

                loss = criterion(y_pred, labels)

                losses.update(loss.item(), bsz)

                # ===== TaskILの正解数をカウント =====
                cls_per_task = opt.cls_per_task  # 例: 10
                correct_batch = 0

                unique_classes = torch.unique(labels)
                for cls in unique_classes:
                    cls = cls.item()
                    task_idx = cls // cls_per_task
                    start = task_idx * cls_per_task
                    end = start + cls_per_task

                    # 現クラスのサンプルだけを抽出
                    mask = (labels == cls)
                    masked_preds = y_pred[mask, start:end]   # 該当タスク範囲のみの出力
                    pred_classes = masked_preds.argmax(1)    # 該当範囲内でargmax → [0~9]
                    true_classes = cls % cls_per_task        # 対応する正解ラベル → 0~9

                    correct_batch += (pred_classes == true_classes).sum().item()

                correct += correct_batch
                total += bsz
            
        # タスクごとの精度と損失を保存
        task_accuracy = 100.0 * correct / total
        all_task_accuracies.append(task_accuracy)
        all_task_losses.append(losses.avg)

        print(f"[Task {taskid}] Loss: {losses.avg:.4f}, Accuracy: {task_accuracy:.2f}%")

    return all_task_accuracies, all_task_losses


def ncm_cclis(model, ncm_loader, val_loader):

    # modelを評価モードに変更
    model.eval()

    # 訓練用（ncm_loader）データから全サンプルの特徴とラベルを集めるリスト
    all_features = []
    all_labels = []

    # 平均特徴の計算
    with torch.no_grad():
        for idx, (images, labels) in enumerate(ncm_loader):

            # gpu上に配置
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            # modelにデータを入力
            # y_pred, features = model(x=images, return_feat=True)
            features = model.encoder(images)

            # 特徴量とラベルを保存
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            
    
    # リスト内のテンソルを連結
    all_features = torch.cat(all_features, dim=0)  # shape: [N, feature_dim]
    all_labels = torch.cat(all_labels, dim=0)

    unique_labels = torch.unique(all_labels)
    class_means = {}  # {クラスラベル: 平均特徴}
    
    
    # 保存してある特徴とラベルをもとに各クラスの平均を計算
    for label in unique_labels:
        
        # 該当クラスのサンプルインデックスを抽出
        idxs = (all_labels == label)
        feats = all_features[idxs]
        
        # サンプルごとに特徴を平均
        mean_feat = feats.mean(dim=0, keepdim=True)  # shape: [1, feature_dim]
        class_means[int(label.item())] = mean_feat
    

    # 辞書のキー（ラベル）が昇順になるようにソートし，平均特徴量を一つのテンソルに変換
    sorted_labels = sorted(class_means.keys())
    means_list = [class_means[l] for l in sorted_labels]
    class_means_tensor = torch.cat(means_list, dim=0)  # shape: [num_classes, feature_dim]
    print("Computed class means for {} classes.".format(class_means_tensor.shape[0]))

    
    # 検証用データの特徴と各クラスの平均特徴を比較し，最も近いクラスに分類する
    total = 0
    correct = 0
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            
            # gpu上に配置
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            # モデルに検証データを入力して特徴を取得
            # y_pred, features = model(x=images, return_feat=True)
            features = model.encoder(images)

            # バッチ内の各サンプル特徴を正規化
            features_norm = F.normalize(features, p=2, dim=1)

            # クラス平均も同様に正規化（デバイス変換も行う）
            class_means_norm = F.normalize(class_means_tensor.to(features.device), p=2, dim=1)

            # 各サンプルと全クラス平均間のコサイン類似度を計算（内積）
            # shape: [batch_size, num_classes]
            cos_sim = torch.mm(features_norm, class_means_norm.t())

            # 各サンプルについて、最も類似度が高いクラス（＝予測ラベル）を求める
            pred_labels = cos_sim.argmax(dim=1)
            
            total += labels.size(0)
            correct += (pred_labels == labels).sum().item()
    
    ncm_acc = correct / total * 100
    # print("NCM Classification Accuracy: {:.2f}%".format(ncm_acc))


    return ncm_acc




