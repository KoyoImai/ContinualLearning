import os
import csv
import logging
import numpy as np
from collections import defaultdict

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

    for idx, (images, labels) in enumerate(train_loader):

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
        
    # 勾配分析
    gradreplay_analysis_ce(opt, model, optimizer, criterion, gradreplay_train_loader, epoch)
    
    return losses.avg, model2



def gradreplay_analysis_ce(opt, model, optimizer, criterion, grad_loader, epoch):

    if not (opt.grad_analysis and (epoch == opt.epochs - 1 or epoch % opt.grad_analysis_freq == 0)):
        return

    path = f"{opt.explog_path}/grad/task{opt.target_task}"
    os.makedirs(path, exist_ok=True)
    grad_log_path = f"{path}/grad_epoch{epoch}_ce_log.csv"
    is_new_file = not os.path.exists(grad_log_path)
    print("grad_log_path: ", grad_log_path)

    grad_sum_dict = defaultdict(float)
    grad_count_dict = defaultdict(int)


    for data in grad_loader:
        inputs, targets = data
        inputs, targets = inputs.cuda(), targets.cuda()

        # モデルの出力を取得
        y_pred = model(inputs)  # shape: [B, num_classes]

        # ラベルごとのインデックスを収集
        label_to_indices = defaultdict(list)
        for i, label in enumerate(targets):
            label_to_indices[label.item()].append(i)

        for label_val, indices in label_to_indices.items():
            if len(indices) == 0:
                continue

            # 対象インデックスだけ取り出して損失を計算
            y_pred_subset = y_pred[indices]
            target_subset = targets[indices]
            loss = criterion(y_pred_subset, target_subset).sum()

            # 勾配計算
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward(retain_graph=True)

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                param_type = name.split('.')[-1]
                layer_name = '.'.join(name.split('.')[:-1])
                grad = param.grad.detach().cpu()

                if grad.dim() == 4:  # Conv
                    grad = grad.view(grad.shape[0], -1)
                    abs_sum = grad.abs().sum(dim=1)
                    for i, g in enumerate(abs_sum):
                        key = (label_val, layer_name, param_type, str([i]))
                        grad_sum_dict[key] += g.item()
                        grad_count_dict[key] += 1

                elif grad.dim() == 2:  # Linear
                    abs_sum = grad.abs().sum(dim=1)
                    for i, g in enumerate(abs_sum):
                        key = (label_val, layer_name, param_type, str([i]))
                        grad_sum_dict[key] += g.item()
                        grad_count_dict[key] += 1

                elif grad.dim() == 1:  # Bias
                    for i, g in enumerate(grad.abs()):
                        key = (label_val, layer_name, param_type, str([i]))
                        grad_sum_dict[key] += g.item()
                        grad_count_dict[key] += 1


    # 最終的に平均をとってCSV出力
    with open(grad_log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(['current task', 'epoch', 'label', 'layer', 'param_type', 'index', 'grad_mean'])

        for key, grad_sum in grad_sum_dict.items():
            count = grad_count_dict[key]
            grad_mean = grad_sum / len(grad_loader.dataset)
            label_val, layer_name, param_type, index_str = key
            writer.writerow([
                opt.target_task,
                epoch,
                label_val,
                layer_name,
                param_type,
                index_str,
                grad_sum,
                grad_mean,
            ])




# 勾配分析のための関数（全てのデータを対象とするならこれを使用，現在タスク＋リプレイバッファのデータを対象とするなら別関数を使用）
def grad_analysis_ce(opt, model, optimizer, criterion, grad_train_loaders, epoch):

    if (opt.grad_analysis and epoch == opt.epochs-1) or (opt.grad_analysis and epoch % opt.grad_analysis_freq == 0):

        grad_log_path = f"{opt.explog_path}/grad_train_log.csv"
        is_new_file = not os.path.exists(grad_log_path)
        print("grad_log_path: ", grad_log_path)

        with open(grad_log_path, mode='a', newline='') as f:

            writer = csv.writer(f)
            if is_new_file:
                writer.writerow(['current task', 'epoch', 'task', 'layer', 'param_type', 'index', 'grad_value'])

            for taskid, loader in enumerate(grad_train_loaders):

                for data in loader:
                    inputs, targets = data
                    inputs, targets = inputs.cuda(), targets.cuda()

                    y_pred = model(inputs)  # ← 修正：元は `images` だった
                    loss = criterion(y_pred, targets).mean()

                    optimizer.zero_grad()
                    model.zero_grad()
                    loss.backward()

                    
                    # 勾配情報をカーネル単位で出力
                    for name, param in model.named_parameters():
                        if param.requires_grad:

                            param_type = name.split('.')[-1]  # パラメータのタイプ（例: weight, bias）
                            layer_name = '.'.join(name.split('.')[:-1])  # レイヤー名
                            grad = param.grad.detach().cpu()
                    

                            if grad.dim() == 4:  # Conv: [out_ch, in_ch, kH, kW]
                                grad = grad.view(grad.shape[0], -1)  # [out_ch, *]
                                abs_sum = grad.abs().sum(dim=1)
                                for i, g in enumerate(abs_sum):
                                    writer.writerow([
                                        opt.target_task,  # 現在のタスク貢献
                                        epoch,
                                        int(targets[0].item()),  # 代表クラス
                                        layer_name,
                                        param_type,
                                        str([i]),  # カーネル index
                                        g.item()
                                    ])

                            elif grad.dim() == 2:  # Linear: [out_dim, in_dim]
                                abs_sum = grad.abs().sum(dim=1)
                                for i, g in enumerate(abs_sum):
                                    writer.writerow([
                                        opt.target_task,  # 現在のタスク
                                        epoch,
                                        int(targets[0].item()),
                                        layer_name,
                                        param_type,
                                        str([i]),  # 出力ユニット index
                                        g.item()
                                    ])

                            elif grad.dim() == 1:  # Bias: [N]
                                for i, g in enumerate(grad.abs()):
                                    writer.writerow([
                                        opt.target_task,  # 現在のタスク
                                        epoch,
                                        int(targets[0].item()),
                                        layer_name,
                                        param_type,
                                        str([i]),
                                        g.item()
                                    ])






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

