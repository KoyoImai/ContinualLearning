import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from util import AverageMeter


logger = logging.getLogger(__name__)



# 損失計算用
cur_features = []
ref_features = []
old_scores = []
new_scores = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs



def train_lucir(opt, model, model2, criterion, optimizer, scheduler, train_loader, val_loader, epoch, T, lamda):

    # modelをtrainモードに変更
    model.train()

    # model2をevalモードに変更
    model2.eval()

    # 学習記録
    losses = AverageMeter()
    correct = 0
    total = 0


    if opt.target_task > 0:
        num_old_classes = model2.head.out_features
        handle_ref_features = model2.head.register_forward_hook(get_ref_features)
        handle_cur_features = model.head.register_forward_hook(get_cur_features)
        handle_old_scores_bs = model.head.fc1.register_forward_hook(get_old_scores_before_scale)
        handle_new_scores_bs = model.head.fc2.register_forward_hook(get_new_scores_before_scale)

    for idx, (images, labels) in enumerate(train_loader):

        # gpu上に配置
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        
        # print("labels.shape: ", labels.shape)
        
        # モデルにデータを入力
        # y_pred, cur_features = model(images, return_feat=True)
        y_pred, _ = model(images, return_feat=True)
        # print("y_pred.shape: ", y_pred.shape)

        # バッチサイズ
        bsz = labels.shape[0]

        if opt.target_task == 0:
            loss = criterion(y_pred, labels)
        else:

            # 過去モデルにデータを入力
            # ref_outputs, ref_features = model2(images, return_feat=True)
            ref_outputs = model2(images)

            # コサイン距離損失
            loss1 = nn.CosineEmbeddingLoss()(cur_features, ref_features.detach(), \
                    torch.ones(images.shape[0]).cuda()) * lamda
            
            # 交差エントロピー損失
            loss2 = criterion(y_pred, labels)

            # マージンランキング損失
            # print("y_pred.sahpe: ", y_pred.shape)
            # print("old_scores.shape: ", old_scores.shape)
            # print("new_scores.shape: ", new_scores.shape)
            outputs_bs = torch.cat((old_scores, new_scores), dim=1)
            assert(outputs_bs.size()==y_pred.size())

            gt_index = torch.zeros(outputs_bs.size()).cuda()
            gt_index = gt_index.scatter(1, labels.view(-1,1), 1).ge(0.5)
            gt_scores = outputs_bs.masked_select(gt_index)

            # 新クラスにおけるtop-K
            # print("num_old_classes: ", num_old_classes)
            # print("outputs_bs[:, num_old_classes:].shape: ", outputs_bs[:, num_old_classes:].shape)
            max_novel_scores = outputs_bs[:, num_old_classes:].topk(opt.K, dim=1)[0]

            hard_index = labels.lt(num_old_classes)
            hard_num = torch.nonzero(hard_index).size(0)

            if  hard_num > 0:
                gt_scores = gt_scores[hard_index].view(-1, 1).repeat(1, opt.K)
                max_novel_scores = max_novel_scores[hard_index]
                assert(gt_scores.size() == max_novel_scores.size())
                assert(gt_scores.size(0) == hard_num)
                #print("hard example gt scores: ", gt_scores.size(), gt_scores)
                #print("hard example max novel scores: ", max_novel_scores.size(), max_novel_scores)
                loss3 = nn.MarginRankingLoss(margin=opt.dist)(gt_scores.view(-1, 1), \
                    max_novel_scores.view(-1, 1), torch.ones(hard_num*opt.K, 1).cuda()) * opt.lw_mr
            else:
                loss3 = torch.zeros(1).cuda()

            loss = loss1 + loss2 + loss3
        

        # update metric
        losses.update(loss.item(), bsz)

        # 最適化ステップ
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = y_pred.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 学習記録の表示
        if opt.target_task > 0:
            if (idx+1) % opt.print_freq == 0 or idx+1 == len(train_loader):
                print('Train: [{0}][{1}/{2}]\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t' \
                    'loss1 {loss1:.3f}\t' 
                    'loss2 {loss2:.3f}\t'
                    'loss3 {loss3:.3f}\t'
                    'acc {correct:.3f}'.format(
                    epoch, idx + 1, len(train_loader), loss=losses, correct=100.*correct/total, loss1=loss1.item(), loss2=loss2.item(), loss3=loss3.item()))
        else:
            if (idx+1) % opt.print_freq == 0 or idx+1 == len(train_loader):
                print('Train: [{0}][{1}/{2}]\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t' \
                    'acc {correct:.3f}'.format(
                    epoch, idx + 1, len(train_loader), loss=losses, correct=100.*correct/total))
    
    
    if opt.target_task > 0:
        print("Removing register_forward_hook")
        handle_ref_features.remove()
        handle_cur_features.remove()
        handle_old_scores_bs.remove()
        handle_new_scores_bs.remove()

    
    
    return losses.avg, model2



def val_lucir(opt, model, model2, criterion, optimizer, scheduler, train_loader, val_loader, epoch):

    
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



def ncm_lucir(model, ncm_loader, val_loader):

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




