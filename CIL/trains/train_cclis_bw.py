
import math
import logging
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from util import AverageMeter, write_csv
from models.resnet_cifar_co2l import LinearClassifier

logger = logging.getLogger(__name__)



def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr_enc = args.warmup_from_enc + p * (args.warmup_to_enc - args.warmup_from_enc)
        lr_prot = args.warmup_from_prot + p * (args.warmup_to_prot - args.warmup_from_prot)
        lr_list = [lr_enc, lr_enc, lr_prot]

        for idx, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_list[idx]
        

def train_cclis_bw(opt, model, model2, criterion, criterion_bw, optimizer, scheduler, train_loader, epoch, subset_sample_num, score_mask):

    # modelをtrainモードに変更
    model.train()

    losses = AverageMeter()
    distill = AverageMeter()

    distill_type = opt.distill_type

    for idx, (two_crops, labels, importance_weight, index) in enumerate(train_loader):

        images = two_crops[0]
        sub_images = two_crops[1]

        # print("two_crops.shape: ", two_crops.shape)
        # print("images.shape: ", images.shape)
        # print("sub_images.shape: ", sub_images.shape)


        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            sub_images = sub_images.cuda(non_blocking=True)
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
        sub_features, _ = model(sub_images)
        # print("features.shape: ", features.shape)
        # print("output.shape: ", output.shape)

        bw_loss = criterion_bw(z1=features, z2=sub_features)
        # print("bw_loss: ", bw_loss)


        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        target_labels = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))

        # ISSupCon
        loss = criterion(output,
                         features, 
                         labels, 
                         importance_weight, 
                         index, 
                         target_labels=target_labels, 
                         sample_num=subset_sample_num, 
                         score_mask=score_mask)
        write_csv(value=loss.item(), path=opt.explog_path, file_name='snce_loss', epoch=epoch)

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
            
                write_csv(value=loss_distill.item(), path=opt.explog_path, file_name='distill_loss', epoch=epoch)

        else:
            raise ValueError("distill type {} is not supported".format(distill_type))

        loss += bw_loss

        # update metric
        losses.update(loss.item(), bsz)

        # 現在の学習率
        current_lr = optimizer.param_groups[0]['lr']

        # 損失の記録
        write_csv(value=loss.item(), path=opt.explog_path, file_name='loss', epoch=epoch)
        write_csv(value=bw_loss.item(), path=opt.explog_path, file_name='bw_loss', epoch=epoch)
    

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

    return losses.avg, model2
