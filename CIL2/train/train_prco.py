
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

from util import AverageMeter, write_csv
from models.resnet_cifar_co2l import LinearClassifier

from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


def adjust_learning_rate_prco(args, optimizer, epoch):
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



def calcurate_efm(opt):


    return None


# =========================
# 訓練用関数部分
# =========================
def train_prco(opt, model, model2, criterion, optimizer, train_loader, epoch):

    # modelをtrainモードに変更
    model.train()

    losses = AverageMeter()
    distill = AverageMeter()

    distill_type = opt.distill_type

    # EFM を使用した蒸留損失
    efm = model.module.efm
    if torch.cuda.is_available() and (efm is not None):
        efm = efm.cuda()

    for idx, data in enumerate(train_loader):

        # 画像などの取得
        images, labels, index = data

        # バッチサイズ
        bsz = labels.shape[0]

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        

        # normalize the prototypes
        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task

            w = model.module.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            model.module.prototypes.weight.copy_(w)

        # warmup処理
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
        encoded, features, output = model(images)
        output = output.T

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # 現在タスクのクラス
        target_labels = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
        # print("target_labels: ", target_labels)


        # ===========================================================================
        # 新しい知識獲得のための損失計算
        # ===========================================================================
        # プロトタイプベースの対照損失
        loss = criterion(output, features, labels, index, target_labels)
        write_csv(loss.item(), opt.result_path, "issupcon_loss", opt.target_task, epoch)


        # ===========================================================================
        # 過去の知識を保持するための蒸留損失
        # ===========================================================================
        if distill_type == "PRD":

            if opt.target_task > 0:

                # バッチに含まれるラベル一覧
                all_labels = torch.unique(labels).view(-1, 1)
                # print("all_labels.shape: ", all_labels.shape)    # all_labels.shape:  torch.Size([4, 1])

                # 過去タスクのラベル一覧
                prev_all_labels = torch.arange(target_labels[0])
                # print("prev_all_labels.shape: ", prev_all_labels.shape)   # prev_all_labels.shape:  torch.Size([2])
                
                # プロトタイプマスクを作成
                # （過去タスクのクラスに対応した出力のみを抽出可能）
                prototypes_mask = torch.scatter(
                    torch.zeros(len(prev_all_labels), opt.n_cls).float(),
                    1,
                    prev_all_labels.view(-1,1),
                    1
                ).to(device)
                # print("prototypes_mask.shape: ", prototypes_mask.shape)   # prototypes_mask.shape:  torch.Size([2, 10])
                # print("prototypes_mask: ", prototypes_mask)
                # prototypes_mask:  tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                #         [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')

                # 過去タスクのサンプルだけを選別するマスク
                labels_mask = labels < min(target_labels)
                # print("labels_mask.shape: ", labels_mask.shape)     # labels_mask.shape:  torch.Size([512])

                
                # ==================================
                # PRD (現在モデルの出力)
                # ==================================
                # 現在モデルで過去クラスに対応したプロトタイプの出力を計算
                sim_prev_task = torch.matmul(prototypes_mask, output)              # output から 過去クラスに対応した出力のみ取り出す
                features1_sim = torch.div(sim_prev_task, opt.current_temp)         # 温度パラメータで除算

                # 数値安定化
                logits_max1, _ = torch.max(features1_sim, dim=0, keepdim=True)
                features1_sim = features1_sim - logits_max1.detach()  # number stability

                row_size = features1_sim.size(0)
                # print("row_size: ", row_size)      # row_size:  2

                # logits を計算
                logits1 = torch.exp(features1_sim) / torch.exp(features1_sim).sum(dim=0, keepdim=True)


                # ==================================
                # PRD (過去モデルの出力)
                # ==================================              
                with torch.no_grad():
                    # 過去モデルで過去クラスに対応したプロトタイプの出力を計算
                    _, _, sim2_prev_task = model2(images)
                    sim2_prev_task = sim2_prev_task.T
                    sim2_prev_task = torch.matmul(prototypes_mask, sim2_prev_task)
                    features2_sim = torch.div(sim2_prev_task, opt.past_temp)

                    # 数値安定化
                    logits_max2, _ = torch.max(features2_sim, dim=0, keepdim=True)
                    features2_sim = features2_sim - logits_max2.detach()

                    # logits を計算
                    logits2 = torch.exp(features2_sim) / torch.exp(features2_sim).sum(dim=0, keepdim=True)
                

                # 蒸留損失を計算（KL-Divergence）
                loss_distill = (-logits2 * torch.log(logits1)).sum(0).mean()
                # print("loss_distill: ", loss_distill)
                write_csv(loss_distill.item(), opt.result_path, "distill_loss", opt.target_task, epoch)
                loss += opt.distill_power * loss_distill
                distill.update(loss_distill.item(), bsz)
        
        # 特徴ドリフトの方向に重み付けして，学習済タスクに重要な方向へとドリフトすることを防ぐ
        elif distill_type == "EFC":
            if opt.target_task > 0:

                # 過去モデルの出力を獲得
                with torch.no_grad():
                    encoded_pre, features_pre, output_pre = model2(images)
                
                D = features.shape[1]
                
                # Projector 出力の差分を計算
                delta = features - features_pre

                # lamda_{EMF} E_{t-1} + \eta I
                M = opt.lambda_efm * efm + opt.eta_efm * torch.eye(D, device=features.device)

                loss_reg = torch.einsum('bi,ij,bj->b', delta, M, delta).mean()
                write_csv(loss_reg.item(), opt.result_path, "reg_loss", opt.target_task, epoch)
                loss += opt.distill_power * loss_reg
                distill.update(loss_reg.item(), bsz)
                print("loss_reg: ", loss_reg)
        

        # encoderの出力でEFC蒸留
        elif distill_type == "EFCv2":
            if opt.target_task > 0:

                # 過去モデルの出力を獲得
                with torch.no_grad():
                    encoded_pre, features_pre, output_pre = model2(images)
                
                # encodedの正規化
                encoded = F.normalize(encoded, dim=1)
                encoded_pre = F.normalize(encoded_pre, dim=1)
                

                D = encoded.shape[1]
                
                # Projector 出力の差分を計算
                delta = encoded - encoded_pre

                # lamda_{EMF} E_{t-1} + \eta I
                M = opt.lambda_efm * efm + opt.eta_efm * torch.eye(D, device=encoded.device)

                loss_reg = torch.einsum('bi,ij,bj->b', delta, M, delta).mean()
                write_csv(loss_reg.item(), opt.result_path, "reg_loss", opt.target_task, epoch)
                loss += opt.distill_power * loss_reg
                distill.update(loss_reg.item(), bsz)
                print("loss_reg: ", loss_reg)


        elif distill_type == "ND":
            if opt.target_task > 0:

                # 過去モデルの出力を獲得
                with torch.no_grad():
                    encoded_pre, features_pre, output_pre = model2(images)
                
                D = features.shape[1]
                
                # Projector 出力の差分を計算
                delta = features - features_pre

                # L2蒸留損失
                loss_distill = (delta ** 2).sum(dim=1).mean()

                write_csv(loss_distill.item(), opt.result_path, "distill_loss", opt.target_task, epoch)
                loss += opt.distill_power * loss_distill
                distill.update(loss_distill.item(), bsz)
                print("loss_distill: ", loss_distill)
        
        
        
        losses.update(loss.item(), bsz)

        # 現在の学習率
        current_lr = optimizer.param_groups[0]['lr']

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 学習記録の表示
        if (idx+1) % opt.print_freq == 0 or idx+1 == len(train_loader):
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'lr {lr:.5f}'.format(
                   epoch, idx + 1, len(train_loader), loss=losses, lr=current_lr))



    return losses.avg, model2




# =========================
# 検証用関数部分
# =========================
def val_prco(opt, model, model2, linear_loader, val_loader, taskil_loaders, knn_train_loaders, epoch):

    # classifierの準備
    classifier = LinearClassifier(name="resnet18", num_classes=opt.n_cls, seed=opt.seed)
    if torch.cuda.is_available():
        classifier = classifier.cuda()


    # classifierのOptimizer
    optimizer = optim.SGD(classifier.parameters(),
                          lr=opt.linear_learning_rate,
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
                features = model.module.encoder(images)
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
                with torch.no_grad():
                    features = model.module.encoder(images)
                output = classifier(features)
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
    all_task_accuracies, all_task_losses = taskil_val_prco(opt, model, classifier, criterion, taskil_loaders)
    all_task_knn_accuracies = knn_val_prco(opt, model, taskil_loaders, knn_train_loaders)
    print("all_task_knn_accuracies: ", all_task_knn_accuracies)

    classil_acc = np.sum(corr)/np.sum(cnt)*100.
    taskil_acc = correct_task/np.sum(cnt)*100.
    return classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier



def taskil_val_prco(opt, model, classifier,  criterion, val_loaders):

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

                # forward
                with torch.no_grad():
                    features = model.module.encoder(images)
                y_pred = classifier(features)

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



def knn_eval(test_embeddings, test_labels, knn_train_embeddings, knn_train_labels, args):
    
    if args.dataset == 'cifar100':
        n_neighbors = 101
    elif args.dataset == 'cifar10':
        n_neighbors = 501
    else:
        assert False
    
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    X_train = to_numpy(knn_train_embeddings)
    y_train = to_numpy(knn_train_labels).ravel()
    X_test  = to_numpy(test_embeddings)
    y_test  = to_numpy(test_labels).ravel()

    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    pred_labels = neigh.fit(X_train, y_train).predict(X_test)

    # knn_acc = np.sum(pred_labels == test_labels) / pred_labels.size
    knn_acc = (pred_labels == y_test).mean()

    return knn_acc



def knn_val_prco(opt, model, val_loaders, train_loaders):

     # modelをevalモードに変更
    model.eval()

    all_task_knn_accuracies = []
    # all_task_losses = [] # KNN評価では直接的な損失は計算しないので削除またはコメントアウト


    with torch.no_grad():

        task_id = 0

        # 各タスクの訓練用データローダーと検証用データローダーを取り出す
        for val_loader, train_loader in zip(val_loaders, train_loaders):

            # 1. 特徴量バンクの構築
            # 訓練用（feat_loader）データから全サンプルの特徴とラベルを集めるリスト
            all_train_features = []
            all_train_labels = []
            all_val_features = []
            all_val_labels = []
            
            # 訓練用データローダーから特徴量とラベルを取得
            for idx, (images, labels) in enumerate(train_loader):
                images = images.float().cuda()
                labels = labels.cuda()
                bsz = labels.shape[0]

                # 特徴量を取得
                features = model.module.encoder(images)

                # 特徴量とラベルを保存
                all_train_features.append(features.cpu())
                all_train_labels.append(labels.cpu())
            
            # 検証用データローダーから特徴量とラベルを取得
            for idx, (images, labels) in enumerate(val_loader):
                images = images.float().cuda()
                labels = labels.cuda()
                bsz = labels.shape[0]
                
                # 特徴量を取得
                with torch.no_grad():
                    features = model.module.encoder(images)

                # 特徴量とラベルを保存
                all_val_features.append(features.cpu())
                all_val_labels.append(labels.cpu())
            
            # リスト内のテンソルを連結
            all_train_features = torch.cat(all_train_features, dim=0)  # shape: [N, feature_dim]
            all_train_labels = torch.cat(all_train_labels, dim=0)
            all_val_features = torch.cat(all_val_features, dim=0)
            all_val_labels = torch.cat(all_val_labels, dim=0)

            # knn分類
            knn_acc = knn_eval(all_val_features, all_val_labels, all_train_features, all_train_labels, opt)
            all_task_knn_accuracies.append(knn_acc)

            task_id += 1
    
    return all_task_knn_accuracies



def ncm_prco(model, ncm_loader, val_loader):

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
            features = model.module.encoder(images)

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
            features = model.module.encoder(images)

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



def val_prco4timnet(opt, model, model2, linear_loader, val_loader, taskil_loaders, epoch):

    # classifierの準備
    classifier = LinearClassifier(name="resnet18", num_classes=opt.n_cls, seed=opt.seed)
    if torch.cuda.is_available():
        classifier = classifier.cuda()


    # classifierのOptimizer
    optimizer = optim.SGD(classifier.parameters(),
                          lr=opt.linear_learning_rate,
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
                features = model.module.encoder(images)
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
                with torch.no_grad():
                    features = model.module.encoder(images)
                output = classifier(features)
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
    all_task_accuracies, all_task_losses = taskil_val_prco(opt, model, classifier, criterion, taskil_loaders)
    # all_task_knn_accuracies = knn_val_cclis(opt, model, taskil_loaders, knn_train_loaders)
    # print("all_task_knn_accuracies: ", all_task_knn_accuracies)

    classil_acc = np.sum(corr)/np.sum(cnt)*100.
    taskil_acc = correct_task/np.sum(cnt)*100.


    return classil_acc, taskil_acc, all_task_accuracies, classifier
    # return classil_acc, taskil_acc, classifier
