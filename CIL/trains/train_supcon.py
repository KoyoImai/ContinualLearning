import logging
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler


from util import AverageMeter
from models.resnet_cifar_co2l import LinearClassifier

logger = logging.getLogger(__name__)





def train_supcon(opt, model, model2, criterion, optimizer, scheduler, train_loader, epoch):

    # modelをtrainモードに変更
    model.train()

    losses = AverageMeter()
    distill = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):

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

        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        loss = criterion(features, labels)

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











