import logging
import numpy as np

import torch
import torch.nn.functional as F

from util import AverageMeter

logger = logging.getLogger(__name__)



def train_er(opt, model, model2, criterion, optimizer, scheduler, train_loader, val_loader, epoch):

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

    return losses.avg, model2


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

