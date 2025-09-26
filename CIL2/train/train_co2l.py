

from collections import defaultdict
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


from train.utils import ncm_classify
from util import AverageMeter, write_csv
from models.resnet_cifar_co2l import LinearClassifier



def train_co2l(opt, model, model2, criterion, optimizer, scheduler, train_loader, epoch):

    # model を trainモードに変更
    model.train()

    losses = AverageMeter()
    distill = AverageMeter()

    criterion_ce = torch.nn.CrossEntropyLoss()


    for idx, data in enumerate(train_loader):

        # 画像とラベルを取得
        images, labels = data

        # バッチサイズ
        bsz = labels.shape[0]

        # ラベルあり2viewの画像を結合
        images = torch.cat([images[0], images[1]], dim=0)
        # print("images.shape: ", images.shape)     # images.shape:  torch.Size([1024, 3, 32, 32])


        # gpu上に配置
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
        

        with torch.no_grad():
            prev_task_mask = labels < opt.target_task * opt.cls_per_task
            prev_task_mask = prev_task_mask.repeat(2)


        # ===========================================================================
        # ラベルありデータの順伝搬
        # ===========================================================================

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
        
        # 特徴量を2viewに分割
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        # print("f1.shape: ", f1.shape, "f2.shape: ", f2.shape)   # f1.shape:  torch.Size([512, 128]) f2.shape:  torch.Size([512, 128])

        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # print("features.shape: ", features.shape)    # eatures.shape:  torch.Size([512, 2, 128])

        loss = criterion(features, labels, target_labels=list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task)))
        # print("loss: ", loss)
        write_csv(loss.item(), opt.result_path, "supcon_loss", opt.target_task, epoch)


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
            write_csv(loss_distill.item(), opt.result_path, "distill_loss", opt.target_task, epoch)
            loss += opt.distill_power * loss_distill
            distill.update(loss_distill.item(), bsz)

        


        # ===========================================================================
        # 後処理
        # ===========================================================================

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
        


    return losses.avg, model2

        






def val_co2l(opt, model, model2, linear_loader, val_loader, taskil_loaders, knn_train_loaders, epoch):

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
                output = classifier(model.module.encoder(images))
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
    all_task_knn_accuracies = knn_val_cclis(opt, model, taskil_loaders, knn_train_loaders)

    classil_acc = np.sum(corr)/np.sum(cnt)*100.
    taskil_acc = correct_task/np.sum(cnt)*100.
    return classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier


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

                y_pred = classifier(model.module.encoder(images))

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
        n_neighbors = 101
    elif args.dataset == "tiny-imagenet":
        n_neighbors = 101
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



def knn_val_cclis(opt, model, val_loaders, train_loaders):

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


# =====================================================================
# 最も近くに位置するncm_loaderに含まれるデータの平均特徴をもとにクラス分類を行う
# =====================================================================
def ncm_co2l(opt, model, ncm_loader, val_loader):

    # modelをevalモードに変更
    model.eval()

    train_features = defaultdict(list)
    train_encoded = defaultdict(list)


    # ==========================================================
    # 訓練用データから平均特徴を計算
    # ==========================================================
    with torch.no_grad():
        for idx, (images, labels) in enumerate(ncm_loader):

            # gpu上に配置
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            # 特徴量を取り出す
            # modelにデータを入力
            features, encoded = model(images, return_feat=True)
            # print("features.shape: ", features.shape)     # projectorの出力：features.shape:  torch.Size([500, 128])
            # print("encoded.shape: ", encoded.shape)       # encoderの出力：encoded.shape:  torch.Size([500, 512])
            
            # features と encoded を格納する
            for feat, enc, lbl in zip(features, encoded, labels):
                train_features[int(lbl.item())].append(feat.detach().cpu())
                train_encoded[int(lbl.item())].append(enc.detach().cpu())
    

    # ==========================================================
    # 各クラスの平均特徴を計算
    # ==========================================================
    class_mean_features = {}
    class_mean_encoded = {}

    for cls in train_features.keys():
        # torch.stack で [N, feature_dim] にまとめ、meanで平均を計算
        class_mean_features[cls] = torch.mean(torch.stack(train_features[cls]), dim=0)
        class_mean_encoded[cls] = torch.mean(torch.stack(train_encoded[cls]), dim=0)
    

    # ==========================================================
    # 検証用データの特徴量を取り出す
    # ==========================================================
    val_features = []
    val_encoded = []
    val_labels = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):

            # gpu上に配置
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            # 特徴量を取り出す
            features, encoded = model(images, return_feat=True)

            # CPUに戻してリストに追加
            val_features.append(features.detach().cpu())
            val_encoded.append(encoded.detach().cpu())
            val_labels.append(labels.detach().cpu())
    
    # 各バッチを結合して1つのテンソルにまとめる
    val_features = torch.cat(val_features, dim=0)   # shape: [num_val_samples, feature_dim]
    val_encoded = torch.cat(val_encoded, dim=0)     # shape: [num_val_samples, encoded_dim]
    val_labels = torch.cat(val_labels, dim=0)       # shape: [num_val_samples]

    print("=== 検証データ ===")
    print("val_features.shape:", val_features.shape)
    print("val_encoded.shape:", val_encoded.shape)
    print("val_labels.shape:", val_labels.shape)

    pred_labels_euclidean, acc_euclidean = ncm_classify(val_encoded, val_labels, class_mean_encoded, metric="euclidean")
    pred_labels_cosine, acc_cosine = ncm_classify(val_encoded, val_labels, class_mean_encoded, metric="cosine")
    print("acc_euclidean: ", acc_euclidean)
    print("acc_cosine: ", acc_cosine)

    return acc_euclidean, acc_cosine








# def ncm_co2l(model, ncm_loader, val_loader):

#     # modelを評価モードに変更
#     model.eval()

#     # 訓練用（ncm_loader）データから全サンプルの特徴とラベルを集めるリスト
#     all_features = []
#     all_labels = []

#     # 平均特徴の計算
#     with torch.no_grad():
#         for idx, (images, labels) in enumerate(ncm_loader):

#             # gpu上に配置
#             if torch.cuda.is_available():
#                 images = images.cuda(non_blocking=True)
#                 labels = labels.cuda(non_blocking=True)
            
#             # modelにデータを入力
#             y_pred, features = model(x=images, return_feat=True)

#             # 特徴量とラベルを保存
#             all_features.append(features.cpu())
#             all_labels.append(labels.cpu())
            
    
#     # リスト内のテンソルを連結
#     all_features = torch.cat(all_features, dim=0)  # shape: [N, feature_dim]
#     all_labels = torch.cat(all_labels, dim=0)

#     unique_labels = torch.unique(all_labels)
#     class_means = {}  # {クラスラベル: 平均特徴}
    
    
#     # 保存してある特徴とラベルをもとに各クラスの平均を計算
#     for label in unique_labels:
        
#         # 該当クラスのサンプルインデックスを抽出
#         idxs = (all_labels == label)
#         feats = all_features[idxs]
        
#         # サンプルごとに特徴を平均
#         mean_feat = feats.mean(dim=0, keepdim=True)  # shape: [1, feature_dim]
#         class_means[int(label.item())] = mean_feat
    

#     # 辞書のキー（ラベル）が昇順になるようにソートし，平均特徴量を一つのテンソルに変換
#     sorted_labels = sorted(class_means.keys())
#     means_list = [class_means[l] for l in sorted_labels]
#     class_means_tensor = torch.cat(means_list, dim=0)  # shape: [num_classes, feature_dim]
#     print("Computed class means for {} classes.".format(class_means_tensor.shape[0]))

    
#     # 検証用データの特徴と各クラスの平均特徴を比較し，最も近いクラスに分類する
#     total = 0
#     correct = 0
#     with torch.no_grad():
#         for idx, (images, labels) in enumerate(val_loader):
            
#             # gpu上に配置
#             if torch.cuda.is_available():
#                 images = images.cuda(non_blocking=True)
#                 labels = labels.cuda(non_blocking=True)
            
#             # モデルに検証データを入力して特徴を取得
#             y_pred, features = model(x=images, return_feat=True)

#             # バッチ内の各サンプル特徴を正規化
#             features_norm = F.normalize(features, p=2, dim=1)

#             # クラス平均も同様に正規化（デバイス変換も行う）
#             class_means_norm = F.normalize(class_means_tensor.to(features.device), p=2, dim=1)

#             # 各サンプルと全クラス平均間のコサイン類似度を計算（内積）
#             # shape: [batch_size, num_classes]
#             cos_sim = torch.mm(features_norm, class_means_norm.t())

#             # 各サンプルについて、最も類似度が高いクラス（＝予測ラベル）を求める
#             pred_labels = cos_sim.argmax(dim=1)
            
#             total += labels.size(0)
#             correct += (pred_labels == labels).sum().item()
    
#     ncm_acc = correct / total * 100
#     # print("NCM Classification Accuracy: {:.2f}%".format(ncm_acc))


#     return ncm_acc






def val_co2l4timnet(opt, model, model2, linear_loader, val_loader, taskil_loaders, knn_train_loaders, epoch):

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
                output = classifier(model.module.encoder(images))
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
    all_task_knn_accuracies = knn_val_cclis(opt, model, taskil_loaders, knn_train_loaders)

    classil_acc = np.sum(corr)/np.sum(cnt)*100.
    taskil_acc = correct_task/np.sum(cnt)*100.
    return classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, classifier