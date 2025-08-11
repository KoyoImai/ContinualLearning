import os
import csv
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn.functional as F

from util import AverageMeter

logger = logging.getLogger(__name__)



def train_er(opt, model, model2, criterion, optimizer, scheduler, train_loader, val_loader, grad_train_loaders, grad_val_loaders,
             gradtask_train_loaders, gradtask_val_loaders, gradreplay_train_loader, gradreplay_val_loader, epoch):

    # trainモードに変更
    model.train()

    # 学習記録
    losses = AverageMeter()

    corr = [0.] * (opt.target_task + 1) * opt.cls_per_task
    cnt  = [0.] * (opt.target_task + 1) * opt.cls_per_task
    correct_task = 0.0

    # 勾配分析（初回エポックのみ）
    if epoch == 1:
        if (opt.grad_analysis and epoch == opt.epochs-1) or (opt.grad_analysis and epoch % opt.grad_analysis_freq == 0):
            gradreplay_analysis_ce(opt, model, optimizer, gradreplay_train_loader, epoch-1)
            # grad_analysis_ce(opt, model, optimizer, criterion, grad_train_loaders, epoch)

    for idx, (images, labels, _) in enumerate(train_loader):

        # gpu上に配置
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # バッチサイズ
        bsz = labels.shape[0]

        # モデルにデータを入力して出力を取得
        y_pred = model(images)
        # print("y_pred.shape: ", y_pred.shape)

        # 損失を計算
        loss = criterion(y_pred, labels).mean()

        # update metric
        losses.update(loss.item(), bsz)


        # 正解率の計算
        cls_list = np.unique(labels.cpu())
        correct_all = (y_pred.argmax(1) == labels)

        for tc in cls_list:
            mask = labels == tc
            correct_task += (y_pred[mask, (tc // opt.cls_per_task) * opt.cls_per_task : ((tc // opt.cls_per_task)+1) * opt.cls_per_task].argmax(1) == (tc % opt.cls_per_task)).float().sum()

        for c in cls_list:
            mask = labels == c
            corr[c] += correct_all[mask].float().sum().item()
            cnt[c] += mask.float().sum().item()

        if idx % opt.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                    'Acc@1 {top1:.3f} {task_il:.3f}'.format(
                        idx, len(val_loader),top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.
                    ))

        # 最適化ステップ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 現在の学習率
        current_lr = optimizer.param_groups[0]['lr']

        # 学習記録の表示
        if (idx+1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1:.3f} {task_il:.3f}\t'
                  'lr {lr:.5f}'.format(
                   epoch, idx + 1, len(train_loader), loss=losses, top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100., lr=current_lr))
        
    # # 勾配分析
    # if (opt.grad_analysis and epoch == opt.epochs-1) or (opt.grad_analysis and epoch % opt.grad_analysis_freq == 0):
    #     gradreplay_analysis_ce(opt, model, optimizer, criterion, gradreplay_train_loader, epoch)
    #     # grad_analysis_ce(opt, model, optimizer, criterion, grad_train_loaders, epoch)

    # 勾配分析
    if (opt.grad_analysis and epoch == opt.epochs-1) or (opt.grad_analysis and epoch % opt.grad_analysis_freq == 0):
        model.eval()
        gradreplay_analysis_ce(opt, model, optimizer, gradreplay_train_loader, epoch)
        model.train()

    return losses.avg, model2








import pandas as pd
from collections import defaultdict

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
CHANNEL_LIMIT = None    # 例: 5 にすると先頭5チャネルだけにスライス


def gradreplay_analysis_ce(opt, model, optimizer, grad_loader, epoch):

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
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    for _, (images, labels, _) in tqdm(enumerate(grad_loader)):


        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            criterion = criterion.cuda()

        device = images.device

        # モデルの出力を取得
        y_pred = model(images)  # shape: [B, num_classes]

        # 
        loss = criterion(y_pred, labels)
        
        # print("y_pred.shape: ", y_pred.shape)   # y_pred.shape:  torch.Size([100, 2])
        # print("loss.shape: ", loss.shape)       # loss.shape:  torch.Size([100])
        
        
        # バッチ内を label ごとにまとめる
        label_to_indices = defaultdict(list)
        for i in range(labels.size(0)):
            label_to_indices[labels[i].item()].append(i)

        # === ラベルごとに backward → 勾配ベクトル化(GPU) → 再構築・集計 ===
        for label_i, indices in label_to_indices.items():

            # label_i の損失のみを取り出して総和を計算
            loss_i = loss[indices].sum()

            # アンカーの数
            n_anchor = len(indices) 

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

            torch.set_printoptions(precision=16, sci_mode=False)

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

                # print("grads[0]: ", grads[0])   # tensor(-0.3077, device='cuda:0')
                # print("grads[0]: ", grads[0])
                # assert False
                

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
                        # grad_count_dict[key] += 1
                        grad_count_dict[key] += n_anchor 
                elif grad_tensor.dim() == 2:
                    abs_sum = grad_tensor.abs().sum(dim=1)
                    for j, g in enumerate(abs_sum.tolist()):
                        key = (label_i, layer_name, param_type, str([j]))
                        grad_sum_dict[key] += g
                        # grad_count_dict[key] += 1
                        grad_count_dict[key] += n_anchor 
                elif grad_tensor.dim() == 1:
                    for j, g in enumerate(grad_tensor.abs().tolist()):
                        key = (label_i, layer_name, param_type, str([j]))
                        grad_sum_dict[key] += g
                        # grad_count_dict[key] += 1
                        grad_count_dict[key] += n_anchor 


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
                # detail_counts[key_li] += 1
                detail_counts[key_li] += n_anchor 
    

    # === ノルムのCSV出力（従来形式） ===
    with open(grad_log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if is_new_log_file:
            writer.writerow(['current task', 'epoch', 'anchor_label', 'layer', 'param_type', 'index', 'grad_count', 'grad_sum', 'grad_mean'])
        for key, grad_sum in grad_sum_dict.items():

            cnt = grad_count_dict[key]

            # 平均計算
            # grad_mean = grad_sum / len(grad_loader.dataset)   # データセット全体で平均を計算
            grad_mean = grad_sum / cnt                         # key の出現回数で平均を計算

            label_i, layer_name, param_type, index_str = key
            writer.writerow([opt.target_task, epoch, label_i, layer_name, param_type, index_str, cnt, grad_sum, grad_mean])

    
    # === 詳細勾配：GPUで合算したテンソルを最後に一括でCPUへ → 行生成して保存 ===
    rows = []
    for (label_i, i), tensor_sum in detail_sums.items():

        # パラメータのレイヤー名などを取り出す
        meta = param_index_map[i]
        layer_name = meta["layer"]; param_type = meta["param_type"]; shape = meta["shape"]

        # カウント
        cnt = detail_counts[(label_i, i)]
        
        # 平均計算と総和取り出し
        # mean_tensor = (tensor_sum / cnt).detach().cpu().reshape(-1)
        sum_tensor = tensor_sum.detach().cpu().reshape(-1)


        # 形状確認
        # print("mean_tensor.shape: ", mean_tensor.shape)   # mean_tensor.shape:  torch.Size([1728])

        # flatten index -> multi-index に戻す
        for flat_idx, g in enumerate(sum_tensor.tolist()):
            idx_tuple = np.unravel_index(flat_idx, shape)

            rows.append([label_i, layer_name, param_type, str(list(idx_tuple)), g])


    write_full_grad_to_csv_from_rows(rows, full_grad_log_path, epoch)




# def gradreplay_analysis_ce(opt, model, optimizer, criterion, grad_loader, epoch):

#     if not (opt.grad_analysis and (epoch == opt.epochs - 1 or epoch % opt.grad_analysis_freq == 0)):
#         return

#     path = f"{opt.explog_path}/gradreplay/task{opt.target_task}"
#     os.makedirs(path, exist_ok=True)
#     grad_log_path = f"{path}/grad_epoch{epoch}_ce_log.csv"
#     is_new_file = not os.path.exists(grad_log_path)
#     print("grad_log_path: ", grad_log_path)

#     grad_sum_dict = defaultdict(float)
#     grad_count_dict = defaultdict(int)


#     for data in grad_loader:
#         inputs, targets = data
#         inputs, targets = inputs.cuda(), targets.cuda()

#         # モデルの出力を取得
#         y_pred = model(inputs)  # shape: [B, num_classes]

#         # ラベルごとのインデックスを収集
#         label_to_indices = defaultdict(list)
#         for i, label in enumerate(targets):
#             label_to_indices[label.item()].append(i)

#         for label_val, indices in label_to_indices.items():
#             if len(indices) == 0:
#                 continue

#             # print("label_val: ", label_val)
#             # print("indices: ", indices)
#             # print()

#             # 対象インデックスだけ取り出して損失を計算
#             y_pred_subset = y_pred[indices]
#             target_subset = targets[indices]
#             loss = criterion(y_pred_subset, target_subset).sum()

#             # 勾配計算
#             optimizer.zero_grad()
#             model.zero_grad()
#             loss.backward(retain_graph=True)

#             for name, param in model.named_parameters():
#                 if not param.requires_grad:
#                     continue

#                 param_type = name.split('.')[-1]
#                 layer_name = '.'.join(name.split('.')[:-1])
#                 grad = param.grad.detach().cpu()

#                 if grad.dim() == 4:  # Conv
#                     grad = grad.view(grad.shape[0], -1)
#                     abs_sum = grad.abs().sum(dim=1)
#                     for i, g in enumerate(abs_sum):
#                         key = (label_val, layer_name, param_type, str([i]))
#                         grad_sum_dict[key] += g.item()
#                         grad_count_dict[key] += 1

#                 elif grad.dim() == 2:  # Linear
#                     abs_sum = grad.abs().sum(dim=1)
#                     for i, g in enumerate(abs_sum):
#                         key = (label_val, layer_name, param_type, str([i]))
#                         grad_sum_dict[key] += g.item()
#                         grad_count_dict[key] += 1

#                 elif grad.dim() == 1:  # Bias
#                     for i, g in enumerate(grad.abs()):
#                         key = (label_val, layer_name, param_type, str([i]))
#                         grad_sum_dict[key] += g.item()
#                         grad_count_dict[key] += 1


#     # 最終的に平均をとってCSV出力
#     with open(grad_log_path, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         if is_new_file:
#             writer.writerow(['current task', 'epoch', 'label', 'layer', 'param_type', 'index', 'grad_sum', 'grad_mean'])

#         for key, grad_sum in grad_sum_dict.items():
#             count = grad_count_dict[key]
#             grad_mean = grad_sum / len(grad_loader.dataset)
#             label_val, layer_name, param_type, index_str = key
#             writer.writerow([
#                 opt.target_task,
#                 epoch,
#                 label_val,
#                 layer_name,
#                 param_type,
#                 index_str,
#                 grad_sum,
#                 grad_mean,
#             ])




# # 勾配分析のための関数（全てのデータを対象とするならこれを使用，現在タスク＋リプレイバッファのデータを対象とするなら別関数を使用）
# def grad_analysis_ce(opt, model, optimizer, criterion, grad_train_loaders, epoch):

#     if (opt.grad_analysis and epoch == opt.epochs-1) or (opt.grad_analysis and epoch % opt.grad_analysis_freq == 0):

#         path = f"{opt.explog_path}/grad/task{opt.target_task}"
#         os.makedirs(path, exist_ok=True)
#         grad_log_path = f"{path}/grad_epoch{epoch}_ce_log.csv"
#         is_new_file = not os.path.exists(grad_log_path)
#         print("grad_log_path: ", grad_log_path)

#         with open(grad_log_path, mode='a', newline='') as f:

#             writer = csv.writer(f)
#             if is_new_file:
#                 writer.writerow(['current task', 'epoch', 'task', 'layer', 'param_type', 'index', 'grad_value'])

#             for taskid, loader in enumerate(grad_train_loaders):

#                 for data in loader:
#                     inputs, targets = data
#                     inputs, targets = inputs.cuda(), targets.cuda()

#                     y_pred = model(inputs)
#                     loss = criterion(y_pred, targets).mean()

#                     optimizer.zero_grad()
#                     model.zero_grad()
#                     loss.backward()

                    
#                     # 勾配情報をカーネル単位で出力
#                     for name, param in model.named_parameters():
#                         if param.requires_grad:

#                             param_type = name.split('.')[-1]  # パラメータのタイプ（例: weight, bias）
#                             layer_name = '.'.join(name.split('.')[:-1])  # レイヤー名
#                             grad = param.grad.detach().cpu()
                    

#                             if grad.dim() == 4:  # Conv: [out_ch, in_ch, kH, kW]
#                                 grad = grad.view(grad.shape[0], -1)  # [out_ch, *]
#                                 abs_sum = grad.abs().sum(dim=1)
#                                 for i, g in enumerate(abs_sum):
#                                     writer.writerow([
#                                         opt.target_task,  # 現在のタスク貢献
#                                         epoch,
#                                         int(targets[0].item()),  # 代表クラス
#                                         layer_name,
#                                         param_type,
#                                         str([i]),  # カーネル index
#                                         g.item()
#                                     ])

#                             elif grad.dim() == 2:  # Linear: [out_dim, in_dim]
#                                 abs_sum = grad.abs().sum(dim=1)
#                                 for i, g in enumerate(abs_sum):
#                                     writer.writerow([
#                                         opt.target_task,  # 現在のタスク
#                                         epoch,
#                                         int(targets[0].item()),
#                                         layer_name,
#                                         param_type,
#                                         str([i]),  # 出力ユニット index
#                                         g.item()
#                                     ])

#                             elif grad.dim() == 1:  # Bias: [N]
#                                 for i, g in enumerate(grad.abs()):
#                                     writer.writerow([
#                                         opt.target_task,  # 現在のタスク
#                                         epoch,
#                                         int(targets[0].item()),
#                                         layer_name,
#                                         param_type,
#                                         str([i]),
#                                         g.item()
#                                     ])






def val_er(opt, model, model2, criterion, optimizer, scheduler, train_loader, val_loader, epoch):

    
    # modelをevalモードに変更
    model.eval()

    # タスク毎の精度を保持
    corr = [0.] * (opt.target_task + 1) * opt.cls_per_task
    cnt  = [0.] * (opt.target_task + 1) * opt.cls_per_task
    correct_task = 0.0

    losses = AverageMeter()

    with torch.no_grad():

        for idx, (images, labels) in enumerate(val_loader):

            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            y_pred = model(images)
            loss = criterion(y_pred, labels)

            losses.update(loss.item(), bsz)

            cls_list = np.unique(labels.cpu())
            correct_all = (y_pred.argmax(1) == labels)

            for tc in cls_list:
                mask = labels == tc
                correct_task += (y_pred[mask, (tc // opt.cls_per_task) * opt.cls_per_task : ((tc // opt.cls_per_task)+1) * opt.cls_per_task].argmax(1) == (tc % opt.cls_per_task)).float().sum()

            for c in cls_list:
                mask = labels == c
                corr[c] += correct_all[mask].float().sum().item()
                cnt[c] += mask.float().sum().item()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Acc@1 {top1:.3f} {task_il:.3f}'.format(
                          idx, len(val_loader),top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.
                      ))
    print(' * Acc@1 {top1:.3f} {task_il:.3f}'.format(top1=np.sum(corr)/np.sum(cnt)*100., task_il=correct_task/np.sum(cnt)*100.))
    
            
    classil_acc = np.sum(corr)/np.sum(cnt)*100.
    taskil_acc = correct_task/np.sum(cnt)*100.
    return classil_acc, taskil_acc


def ncm_er(model, ncm_loader, val_loader):

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
            y_pred, features = model(x=images, return_feat=True)

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
            y_pred, features = model(x=images, return_feat=True)

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


def taskil_val_er(opt, model, criterion, val_loaders):

    # modelをevalモードに変更
    model.eval()

    all_task_accuracies = []
    all_task_losses = []

    for taskid, val_loader in enumerate(val_loaders):

        if taskid <= opt.target_task:

            losses = AverageMeter()
            correct = 0
            total = 0
            task_accuracy = 0

            with torch.no_grad():

                for idx, (images, labels) in enumerate(val_loader):

                    images = images.float().cuda()
                    labels = labels.cuda()
                    bsz = labels.shape[0]

                    y_pred = model(images)

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
    else:
        print("complete taskil val")

    return all_task_accuracies, all_task_losses


# 出力範囲を制限せずに精度を測っているので，厳密にはtaskilの精度測定ではない
# def taskil_val_er(opt, model, criterion, val_loaders):

#     # modelをevalモードに変更
#     model.eval()

#     all_task_accuracies = []
#     all_task_losses = []

#     for taskid, val_loader in enumerate(val_loaders):

#         losses = AverageMeter()
#         correct = 0
#         total = 0
#         task_accuracy = 0

#         with torch.no_grad():

#             for idx, (images, labels) in enumerate(val_loader):

#                 images = images.float().cuda()
#                 labels = labels.cuda()
#                 bsz = labels.shape[0]

#                 y_pred = model(images)
#                 loss = criterion(y_pred, labels)

#                 losses.update(loss.item(), bsz)

#                 # 正解予測数カウント
#                 preds = y_pred.argmax(1)
#                 correct += (preds == labels).sum().item()
#                 total += bsz
            
#         # タスクごとの精度と損失を保存
#         task_accuracy = 100.0 * correct / total
#         all_task_accuracies.append(task_accuracy)
#         all_task_losses.append(losses.avg)

#         print(f"[Task {taskid}] Loss: {losses.avg:.4f}, Accuracy: {task_accuracy:.2f}%")

#     return all_task_accuracies, all_task_losses

