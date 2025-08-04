import os
import csv
import logging
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from util import AverageMeter
from models.resnet_cifar_co2l import LinearClassifier

logger = logging.getLogger(__name__)


def train_co2l(opt, model, model2, criterion, optimizer, scheduler, train_loader, grad_train_loaders, grad_val_loaders, 
               gradtask_train_loaders, gradtask_val_loaders, gradreplay_train_loader, gradreplay_val_loader, epoch):

    # modelをtrainモードに変更
    model.train()

    losses = AverageMeter()
    distill = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):

        # 2viewの画像を結合
        images = torch.cat([images[0], images[1]], dim=0)
        
        # gpu上に配置
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task
            prev_task_mask = prev_task_mask.repeat(2)

        # modelにデータを入力
        features, encoded = model(images, return_feat=True)

        # IRD損失の一部（現在タスク）を計算
        if opt.target_task > 0:
            features1_prev_task = features

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
            # print("logits1.shape: ", logits1.shape)  # logits1.shape:  torch.Size([1024, 1023])
            

        # 特徴量を2viewに分割
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        # print("f1.shape: ", f1.shape, "f2.shape: ", f2.shape)   # f1.shape:  torch.Size([512, 128]) f2.shape:  torch.Size([512, 128])

        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # print("features.shape: ", features.shape)    # eatures.shape:  torch.Size([512, 2, 128])

        loss = criterion(features, labels, target_labels=list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task)))

        # IRD損失の一部（過去タスク）を計算
        if opt.target_task > 0:
            with torch.no_grad():
                features2_prev_task = model2(images)

                features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), opt.past_temp)
                logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                features2_sim = features2_sim - logits_max2.detach()
                logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)
                # print('logits2.shape: ', logits2.shape)  # logits2.shape:  torch.Size([1024, 1023])

            # print("(-logits2 * torch.log(logits1)).shape: ", (-logits2 * torch.log(logits1)).shape)                # (-logits2 * torch.log(logits1)).shape:  torch.Size([1024, 1023])
            # print("(-logits2 * torch.log(logits1)).sum(1).shape: ", (-logits2 * torch.log(logits1)).sum(1).shape)  # (-logits2 * torch.log(logits1)).sum(1).shape:  torch.Size([1024])
            loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
            loss += opt.distill_power * loss_distill
            distill.update(loss_distill.item(), bsz)


        # update metric
        losses.update(loss.item(), bsz)

        # 現在の学習率
        current_lr = optimizer.param_groups[0]['lr']

        # 最適化ステップ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 学習記録の表示
        if (idx+1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'lr {lr:.5f}'.format(
                   epoch, idx + 1, len(train_loader), loss=losses, lr=current_lr))
            

    # 勾配分析（訓練用）
    if (opt.grad_analysis and epoch == opt.epochs-1) or (opt.grad_analysis and epoch % opt.grad_analysis_freq == 0):
        grad_analysis_supcon(opt, model, optimizer, criterion, gradreplay_train_loader, epoch)
        if opt.target_task > 0:
            grad_analysis_distill(opt, model, model2, optimizer, criterion, gradreplay_train_loader, epoch)
        
    return losses.avg, model2






from collections import defaultdict
def grad_analysis_supcon(opt, model, optimizer, criterion, grad_loader, epoch):
    if not (opt.grad_analysis and (epoch == opt.epochs - 1 or epoch % opt.grad_analysis_freq == 0)):
        return

    path = f"{opt.explog_path}/gradreplay/task{opt.target_task}"
    os.makedirs(path, exist_ok=True)
    grad_log_path = f"{path}/grad_epoch{epoch}_supcon_log.csv"
    is_new_file = not os.path.exists(grad_log_path)
    print("grad_log_path: ", grad_log_path)

    # 勾配の合計値と回数の辞書（平均を出すため）
    grad_sum_dict = defaultdict(float)
    grad_count_dict = defaultdict(int)


    for (images, labels) in grad_loader:

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]
        features, _ = model(images, return_feat=True)

        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss_tensor = criterion(features, labels,
                                reduction='grad_analysis',
                                target_labels=list(range(opt.target_task * opt.cls_per_task,
                                                            (opt.target_task + 1) * opt.cls_per_task)))  # [2, bsz]

        # ラベルごとにサンプルをまとめる
        label_to_indices = defaultdict(list)
        
        for i in range(bsz):
            label = labels[i].item()
            label_to_indices[label].append(i)

        for label_i, indices in label_to_indices.items():
            # 指定ラベルの損失を平均して backward
            # loss_i = loss_tensor[:, indices].mean()
            loss_i = loss_tensor[:, indices].sum()
            optimizer.zero_grad()
            model.zero_grad()
            loss_i.backward(retain_graph=True)

            for name, param in model.named_parameters():
                if not param.requires_grad:
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

    #  最終的に平均値をCSVに書き出し
    with open(grad_log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(['current task', 'epoch', 'anchor_label', 'layer', 'param_type', 'index', 'grad_sum', 'grad_mean'])

        for key, grad_sum in grad_sum_dict.items():
            count = grad_count_dict[key]
            # grad_mean = grad_sum / count if count > 0 else 0.0
            grad_mean = grad_sum / 2*len(grad_loader.dataset)
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




def grad_analysis_distill(opt, model, model2, optimizer, criterion, grad_loader, epoch):
    if not (opt.grad_analysis and (epoch == opt.epochs - 1 or epoch % opt.grad_analysis_freq == 0)):
        return

    path = f"{opt.explog_path}/gradreplay/task{opt.target_task}"
    os.makedirs(path, exist_ok=True)
    grad_log_path = f"{path}/grad_epoch{epoch}_distill_log.csv"
    is_new_file = not os.path.exists(grad_log_path)
    print("grad_log_path: ", grad_log_path)

    # 勾配の合計値と回数の辞書（平均を出すため）
    grad_sum_dict = defaultdict(float)
    grad_count_dict = defaultdict(int)


    for (images, labels) in grad_loader:

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        bsz = labels.shape[0]
        
        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task
            prev_task_mask = prev_task_mask.repeat(2)

        # modelにデータを入力
        features, encoded = model(images, return_feat=True)

        # IRD損失の一部（現在タスク）を計算
        if opt.target_task > 0:
            features1_prev_task = features

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
            # print("logits1.shape: ", logits1.shape)  # logits1.shape:  torch.Size([1024, 1023])


        # IRD損失の一部（過去タスク）を計算
        if opt.target_task > 0:
            with torch.no_grad():
                features2_prev_task = model2(images)

                features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), opt.past_temp)
                logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
                features2_sim = features2_sim - logits_max2.detach()
                logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)


            # print("(-logits2 * torch.log(logits1)).shape: ", (-logits2 * torch.log(logits1)).shape)                # (-logits2 * torch.log(logits1)).shape:  torch.Size([1024, 1023])
            # print("(-logits2 * torch.log(logits1)).sum(1).shape: ", (-logits2 * torch.log(logits1)).sum(1).shape)  # (-logits2 * torch.log(logits1)).sum(1).shape:  torch.Size([1024])
            loss_distill = (-logits2 * torch.log(logits1)).sum(1)
            loss_tensor = opt.distill_power * loss_distill



        # ラベルごとにサンプルをまとめる
        label_to_indices = defaultdict(list)

        # ラベルとそのインデックス
        for i in range(2 * bsz):
            label = labels[i % bsz].item()
            label_to_indices[label].append(i)

        for label_i, indices in label_to_indices.items():
            # loss_i = loss_tensor[indices].mean()
            loss_i = loss_tensor[indices].sum()

            optimizer.zero_grad()
            model.zero_grad()
            loss_i.backward(retain_graph=True)

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                param_type = name.split('.')[-1]
                layer_name = '.'.join(name.split('.')[:-1])
                grad = param.grad.detach().cpu()

                if grad.dim() == 4:  # Conv: [out_ch, in_ch, kH, kW]
                    grad = grad.view(grad.shape[0], -1)
                    abs_sum = grad.abs().sum(dim=1)
                    for j, g in enumerate(abs_sum):
                        key = (label_i, layer_name, param_type, str([j]))
                        grad_sum_dict[key] += g.item()
                        grad_count_dict[key] += 1

                elif grad.dim() == 2:  # Linear: [out_dim, in_dim]
                    abs_sum = grad.abs().sum(dim=1)
                    for j, g in enumerate(abs_sum):
                        key = (label_i, layer_name, param_type, str([j]))
                        grad_sum_dict[key] += g.item()
                        grad_count_dict[key] += 1

                elif grad.dim() == 1:  # Bias
                    for j, g in enumerate(grad.abs()):
                        key = (label_i, layer_name, param_type, str([j]))
                        grad_sum_dict[key] += g.item()
                        grad_count_dict[key] += 1
        


    #  最終的に平均値をCSVに書き出し
    with open(grad_log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(['current task', 'epoch', 'anchor_label', 'layer', 'param_type', 'index', 'grad_sum', 'grad_mean'])

        for key, grad_sum in grad_sum_dict.items():
            count = grad_count_dict[key]
            # grad_mean = grad_sum / count if count > 0 else 0.0
            grad_mean = grad_sum / 2*len(grad_loader.dataset)
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





# ## 一応動く改良版v1（ただしめっちゃ処理が遅い．半日たっても15エポックとか）
# from collections import defaultdict
# def grad_analysis_supcon(opt, model, optimizer, criterion, grad_loaders, epoch):
#     if not (opt.grad_analysis and (epoch == opt.epochs - 1 or epoch % opt.grad_analysis_freq == 0)):
#         return

#     grad_log_path = f"{opt.explog_path}/gradtask_train_supcon_log.csv"
#     is_new_file = not os.path.exists(grad_log_path)
#     print("grad_log_path: ", grad_log_path)

#     # 勾配蓄積用辞書：キー=(label, layer_name, param_type, unit_index)
#     grad_accumulator = defaultdict(float)


#     for taskid, loader in enumerate(grad_loaders):

#         for (images, labels) in loader:
#             images = torch.cat([images[0], images[1]], dim=0)
#             if torch.cuda.is_available():
#                 images = images.cuda(non_blocking=True)
#                 labels = labels.cuda(non_blocking=True)

#             bsz = labels.shape[0]
#             features, _ = model(images, return_feat=True)

#             # 特徴量整形（[B, 2, D]）
#             f1, f2 = torch.split(features, [bsz, bsz], dim=0)
#             features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

#             # 全 anchor の損失を取得（reduction='grad_analysis' は custom reduction='none' と同等）
#             loss = criterion(features, labels,
#                              reduction='grad_analysis',
#                              target_labels=list(range(opt.target_task * opt.cls_per_task,
#                                                       (opt.target_task + 1) * opt.cls_per_task)))  # [2, B]

#             for i in range(loss.shape[1]):  # bsz 回ループ
#                 loss_i = loss[:, i].mean()
#                 label_i = labels[i].item()

#                 optimizer.zero_grad()
#                 model.zero_grad()
#                 loss_i.backward(retain_graph=True)

#                 for name, param in model.named_parameters():
#                     if not param.requires_grad:
#                         continue

#                     param_type = name.split('.')[-1]
#                     layer_name = '.'.join(name.split('.')[:-1])
#                     grad = param.grad.detach().cpu()

#                     if grad.dim() == 4:  # Conv2d: [out_ch, in_ch, kH, kW]
#                         grad = grad.view(grad.shape[0], -1)
#                         abs_sum = grad.abs().sum(dim=1)
#                         for j, g in enumerate(abs_sum):
#                             key = (label_i, layer_name, param_type, str([j]))
#                             grad_accumulator[key] += g.item()

#                     elif grad.dim() == 2:  # Linear: [out_dim, in_dim]
#                         abs_sum = grad.abs().sum(dim=1)
#                         for j, g in enumerate(abs_sum):
#                             key = (label_i, layer_name, param_type, str([j]))
#                             grad_accumulator[key] += g.item()

#                     elif grad.dim() == 1:  # Bias: [N]
#                         for j, g in enumerate(grad.abs()):
#                             key = (label_i, layer_name, param_type, str([j]))
#                             grad_accumulator[key] += g.item()

#     # 勾配集約結果をCSVに一括書き込み
#     with open(grad_log_path, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         if is_new_file:
#             writer.writerow(['current task', 'epoch', 'anchor_label', 'layer', 'param_type', 'index', 'grad_value'])

#         for (label_i, layer_name, param_type, index_str), grad_val in grad_accumulator.items():
#             writer.writerow([
#                 opt.target_task,
#                 epoch,
#                 label_i,
#                 layer_name,
#                 param_type,
#                 index_str,
#                 grad_val
#             ])





# 雛形プログラム（これだけだと動かない，ただの改良用）
# def grad_analysis_supcon(opt, model, optimizer, criterion, grad_loaders, epoch):

#     if (opt.grad_analysis and epoch == opt.epochs-1) or (opt.grad_analysis and epoch % opt.grad_analysis_freq == 0):

#         grad_log_path = f"{opt.explog_path}/gradtask_train_supcon_log.csv"
#         is_new_file = not os.path.exists(grad_log_path)
#         print("grad_log_path: ", grad_log_path)

#         with open(grad_log_path, mode='a', newline='') as f:

#             writer = csv.writer(f)
#             if is_new_file:
#                 writer.writerow(['current task', 'epoch', 'anchor_label', 'layer', 'param_type', 'index', 'grad_value'])

#             for taskid, loader in enumerate(grad_loaders):

#                 # # 勾配の初期化
#                 # optimizer.zero_grad()
#                 # model.zero_grad()

#                 # データローダーに含まれる全てのバッチに対して損失を計算
#                 for (images, labels) in loader:

#                     images = torch.cat([images[0], images[1]], dim=0)
#                     if torch.cuda.is_available():
#                         images = images.cuda(non_blocking=True)
#                         labels = labels.cuda(non_blocking=True)
                    
#                     bsz = labels.shape[0]

#                     # modelにデータを入力
#                     features, encoded = model(images, return_feat=True)

#                     # 非対称な教師あり対照損失
#                     f1, f2 = torch.split(features, [bsz, bsz], dim=0)
#                     features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
#                     loss = criterion(features, labels,
#                                      reduction='grad_analysis',
#                                      target_labels=list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task)))
#                     # print("loss.shape: ", loss.shape)  # loss.shape:  torch.Size([2, 2000])


#                     for i in range(loss.shape[1]):
                        
#                         # サンプルiの損失を取り出す
#                         loss_i = loss[:, i].mean()

#                         # サンプルiのラベルを取り出す
#                         label_i = labels[i].item()
                        
#                         # 勾配を計算
#                         loss_i.backward()
            
#                         # 勾配情報をカーネル単位で出力
#                         for name, param in model.named_parameters():
#                             if param.requires_grad:

#                                 param_type = name.split('.')[-1]  # パラメータのタイプ（例: weight, bias）
#                                 layer_name = '.'.join(name.split('.')[:-1])  # レイヤー名
#                                 grad = param.grad.detach().cpu()
                        

                                
#                                 # ここで勾配の絶対値をcsvファイルに書き込むのではなく，各パラメータの絶対値の合計を計算して辞書に格納
#                                 if grad.dim() == 4:  # Conv: [out_ch, in_ch, kH, kW]
#                                     grad = grad.view(grad.shape[0], -1)  # [out_ch, *]
#                                     abs_sum = grad.abs().sum(dim=1)
#                                     for i, g in enumerate(abs_sum):
#                                         writer.writerow([
#                                             opt.target_task,  # 現在のタスク
#                                             epoch,
#                                             label_i,  # アンカーのラベル
#                                             layer_name,
#                                             param_type,
#                                             str([i]),  # カーネル index
#                                             g.item()
#                                         ])

#                                 elif grad.dim() == 2:  # Linear: [out_dim, in_dim]
#                                     abs_sum = grad.abs().sum(dim=1)
#                                     for i, g in enumerate(abs_sum):
#                                         writer.writerow([
#                                             opt.target_task,  # 現在のタスク
#                                             epoch,
#                                             label_i,  # アンカーのラベル
#                                             layer_name,
#                                             param_type,
#                                             str([i]),  # 出力ユニット index
#                                             g.item()
#                                         ])

#                                 elif grad.dim() == 1:  # Bias: [N]
#                                     for i, g in enumerate(grad.abs()):
#                                         writer.writerow([
#                                             opt.target_task,  # 現在のタスク
#                                             epoch,
#                                             label_i,  # アンカーのラベル
#                                             layer_name,
#                                             param_type,
#                                             str([i]),
#                                             g.item()
#                                         ])



# def grad_analysis_distill(opt, model, model2, optimizer, criterion, grad_loaders, epoch):

#     if (opt.grad_analysis and epoch == opt.epochs-1) or (opt.grad_analysis and epoch % opt.grad_analysis_freq == 0):

#         grad_log_path = f"{opt.explog_path}/gradtask_train_distill_log.csv"
#         is_new_file = not os.path.exists(grad_log_path)
#         print("grad_log_path: ", grad_log_path)

#         with open(grad_log_path, mode='a', newline='') as f:

#             writer = csv.writer(f)
#             if is_new_file:
#                 writer.writerow(['current task', 'epoch', 'task', 'layer', 'param_type', 'index', 'grad_value'])

#             for taskid, loader in enumerate(grad_loaders):

#                 # 勾配の初期化
#                 optimizer.zero_grad()
#                 model.zero_grad()

#                 for (images, labels) in loader:

#                     images = torch.cat([images[0], images[1]], dim=0)
#                     if torch.cuda.is_available():
#                         images = images.cuda(non_blocking=True)
#                         labels = labels.cuda(non_blocking=True)
#                     bsz = labels.shape[0]

#                     with torch.no_grad():
#                         prev_task_mask = labels < opt.target_task * opt.cls_per_task
#                         prev_task_mask = prev_task_mask.repeat(2)

#                     # modelにデータを入力
#                     features, encoded = model(images, return_feat=True)

#                     # IRD損失の一部（現在タスク）を計算
#                     if opt.target_task > 0:
#                         features1_prev_task = features

#                         features1_sim = torch.div(torch.matmul(features1_prev_task, features1_prev_task.T), opt.current_temp)
#                         logits_mask = torch.scatter(
#                             torch.ones_like(features1_sim),
#                             1,
#                             torch.arange(features1_sim.size(0)).view(-1, 1).cuda(non_blocking=True),
#                             0
#                         )
#                         logits_max1, _ = torch.max(features1_sim * logits_mask, dim=1, keepdim=True)
#                         features1_sim = features1_sim - logits_max1.detach()
#                         row_size = features1_sim.size(0)
#                         logits1 = torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)) / torch.exp(features1_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)

#                     # IRD損失の一部（過去タスク）を計算
#                     if opt.target_task > 0:
#                         with torch.no_grad():
#                             features2_prev_task = model2(images)

#                             features2_sim = torch.div(torch.matmul(features2_prev_task, features2_prev_task.T), opt.past_temp)
#                             logits_max2, _ = torch.max(features2_sim*logits_mask, dim=1, keepdim=True)
#                             features2_sim = features2_sim - logits_max2.detach()
#                             logits2 = torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)) /  torch.exp(features2_sim[logits_mask.bool()].view(row_size, -1)).sum(dim=1, keepdim=True)


#                         loss_distill = (-logits2 * torch.log(logits1)).sum(1).mean()
#                         loss = opt.distill_power * loss_distill
                    
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





def val_co2l(opt, model, model2, linear_loader, val_loader, taskil_loaders, epoch):

    # classifierの準備
    classifier = LinearClassifier(name="resnet18", num_classes=opt.n_cls, seed=opt.seed)
    if torch.cuda.is_available():
        classifier = classifier.cuda()
    
    # classifierのOptimizer
    optimizer = optim.SGD(classifier.parameters(),
                          lr=opt.linear_lr,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

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
    all_task_accuracies, all_task_losses = taskil_val_co2l(opt, model, classifier, criterion, taskil_loaders)

    classil_acc = np.sum(corr)/np.sum(cnt)*100.
    taskil_acc = correct_task/np.sum(cnt)*100.
    return classil_acc, taskil_acc, all_task_accuracies, all_task_losses


def taskil_val_co2l(opt, model, classifier,  criterion, val_loaders):

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


def ncm_co2l(model, ncm_loader, val_loader):

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













