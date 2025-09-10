
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




def train_prco(opt, model, model2, criterion, optimizer, train_loader, epoch):

    # modelをtrainモードに変更
    model.train()

    losses = AverageMeter()
    distill = AverageMeter()

    distill_type = opt.distill_type

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
        
        features, output = model(images)
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
                    _, sim2_prev_task = model2(images)
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
        
        elif distill_type == "EFCD":            
            assert False
        
        else:
            assert False

        
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











