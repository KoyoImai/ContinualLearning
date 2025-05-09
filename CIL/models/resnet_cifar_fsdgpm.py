import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms

import os
import os.path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import random
import pdb
import argparse,time
import math
from copy import deepcopy

from util import seed_everything



## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=1, bias=False)



# 外部パラメータによる順伝播用のヘルパー関数
def basicblock_forward(x, block, params, idx):
    """
    BasicBlock の関数的順伝播。
    params: 外部パラメータリス
    idx: 現在のパラメータ位置。
    順序（shortcut が不要な場合）：
      [conv1.weight, bn1.weight, bn1.bias, conv2.weight, bn2.weight, bn2.bias]
    shortcut が必要な場合はさらに [shortcut.conv.weight, shortcut.bn.weight, shortcut.bn.bias] を追加。
    """
    
    # conv1
    weight_conv1 = params[idx]
    idx += 1
    weight_bn1 = params[idx]
    idx += 1
    bias_bn1 = params[idx]
    idx += 1
    out = F.conv2d(x, weight_conv1, bias=None, stride=block.conv1.stride, padding=block.conv1.padding)
    out = F.batch_norm(out, block.bn1.running_mean, block.bn1.running_var,
                       weight=weight_bn1, bias=bias_bn1, training=True)
    out = F.relu(out)

    # conv2
    weight_conv2 = params[idx]
    idx += 1
    weight_bn2 = params[idx]
    idx += 1
    bias_bn2 = params[idx]
    idx += 1
    out = F.conv2d(out, weight_conv2, bias=None, stride=block.conv2.stride, padding=block.conv2.padding)
    out = F.batch_norm(out, block.bn2.running_mean, block.bn2.running_var,
                       weight=weight_bn2, bias=bias_bn2, training=True)
    # shortcut
    if len(block.shortcut) > 0:
        weight_short = params[idx]
        idx += 1
        weight_short_bn = params[idx]
        idx += 1
        bias_short_bn = params[idx]
        idx += 1
        shortcut = F.conv2d(x, weight_short, bias=None, stride=block.shortcut[0].stride, padding=0)
        shortcut = F.batch_norm(shortcut, block.shortcut[1].running_mean, block.shortcut[1].running_var,
                                weight=weight_short_bn, bias=bias_short_bn, training=True)
    else:
        shortcut = x
    
    out = out + shortcut
    out = F.relu(out)
    
    return out, idx


def functional_layer_forward(x, layer, params, idx):
    """
    指定された Sequential で構成された層（layer1～layer4）の各 BasicBlock について，
    外部パラメータを用いた順伝播を行う．
    """
    for block in layer:
        x, idx = basicblock_forward(x, block, params, idx)
    return x, idx


# バッチ正規化のあれこれを変更したバージョン
class BasicBlock(nn.Module):
    expansion = 1
    # expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)  # 元々のコード
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
                nn.BatchNorm2d(self.expansion * planes)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x
        self.count +=1
        out = relu(self.bn1(self.conv1(x)))
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nf=64, nclass=100, seed=777, opt=None, zero_init_residual=False):
        super(ResNet, self).__init__()
        
        seed_everything(seed=seed)

        self.freeze_bn = opt.freeze_bn
        self.second_order = opt.second_order
        self.opt = opt

        if opt.dataset == "tiny-imagenet":
            self.n_rep = 21
        else:
            self.n_rep = 21

        # eta1: 重みへのせちつどうの更新ステップサイズ
        self.eta1 = opt.eta1

        # リプレイバッファの設定
        self.memories = opt.mem_size  # mem_size of M
        self.age = 0  # total number of training samples
        self.M = []

        # setup GPM
        self.mem_batch_size = opt.mem_batch_size
        self.M_vec = []    # bases of GPM
        self.M_val = []    # eigenvalues of each basis of GPM
        self.M_task = []   # the task id of each basis, only used to analysis GPM

        # setup losses
        self.loss_ce = torch.nn.CrossEntropyLoss()

        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1, stride=1)
        # self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.bn1 = nn.BatchNorm2d(nf * 1)


        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # <- 追加部分
        
        # self.linear=torch.nn.ModuleList()
        # self.fc = nn.Linear(nf * 8 * block.expansion * 4, nclass, bias=False)   # 元コード
        self.fc = nn.Linear(nf * 8 * block.expansion, nclass, bias=True)
        self.act = OrderedDict()

        # パラメータ初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x, return_feat=False):
        bsz = x.size(0)

        # self.act['conv_in'] = x.view(bsz, 3, 32, 32)
        # out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32)))) 

        self.act['conv_in'] = x.view(bsz, 3, 64, 64)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 64, 64))))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = out
        
        # out = avg_pool2d(out, 2)   # 元々はこれ
        out = out.view(out.size(0), -1)

        if return_feat:
            return self.fc(out), feat
        
        return self.fc(out)


    def get_params(self):
        self.vars = []
        for p in list(self.parameters()):
            if p.requires_grad:
                self.vars.append(p)
        return self.vars
    

    def conv_to_linear(self, x, conv, batchsize=None):
        
        kernel = conv.kernel_size
        stride = conv.stride
        padding = conv.padding

        if batchsize is None:
            batchsize = x.size(0)
        
        # パディングが必要な場合
        if padding[0] > 0 or padding[1] > 0:
            y = torch.zeros((x.size(0), x.size(1), x.size(2) + 2*padding[0],
                             x.size(3) + 2*padding[1]), device=x.device, dtype=x.dtype)
            y[:, :, padding[0]:padding[0]+x.size(2), padding[1]:padding[1]+x.size(3)] = x
        else:
            y = x
        
        # 形状獲得
        h = y.size(2)
        w = y.size(3)
        kh, kw = kernel[0], kernel[1]
        fs = []

        # スライディングウィンドウで各局所領域を抽出
        for i in range(0, h, stride[0]):
            for j in range(0, w, stride[1]):
                if i+kh > h or j+kw > w:
                    break
                f = y[:, :, i:i+kh, j:j+kw]
                # f = f.view(batchsize, 1, -1)   # 各パッチを1次元にflatten
                f = f.reshape(batchsize, 1, -1)  # 各パッチを1次元にflatten
                if i == 0 and j == 0:
                    fs = f
                else:
                    fs = torch.cat((fs, f), dim=1)
        
        # 最終的に (batchsize * num_patches, channels*kh*kw) の形状に変換
        fs = fs.view(-1, fs.size(-1))
        
        # conv 層そのものの出力
        h_out = conv(x)
        
        return fs, h_out


    def svd_block_forward(self, block, x, y):
        
        """
        'block' を通す際に、conv1とconv2の入力に対して
        conv_to_linear() でパッチ行列を取得し、リスト y に保存する。
        最後に block の出力を返す。
        """
        
        # ----- conv1 -----
        # conv1 へ入力される x に対してパッチを生成
        fs_conv1, conv1_in = self.conv_to_linear(x, block.conv1)
        y.append(fs_conv1)  
        # conv1 -> bn1 -> relu
        # print("conv1_in.shape: ", conv1_in.shape)
        # out = block.conv1(conv1_in)
        out = block.conv1(x)
        out = block.bn1(out)
        out = F.relu(out)

        # ----- conv2 -----
        # conv2 へ入力される out に対してパッチを生成
        fs_conv2, conv2_in = self.conv_to_linear(out, block.conv2)
        y.append(fs_conv2)
        # conv2 -> bn2
        # out2 = block.conv2(conv2_in)
        out2 = block.conv2(out)
        out2 = block.bn2(out2)

        # shortcut
        if len(block.shortcut) > 0:
            # sc = block.shortcut(x)
            shortcut_conv = block.shortcut[0]
            shortcut_bn = block.shortcut[1]

            fs_sc, sc_in = self.conv_to_linear(x, shortcut_conv)
            # y.append(fs_sc)  # shortcut conv 入力
            y.append(fs_sc)    # shortcut conv 入力
            sc_out = shortcut_conv(x)
            sc_out = shortcut_bn(sc_out)
        else:
            sc_out = x

        out = out2 + sc_out
        out = F.relu(out)

        return out

    def net_forward(self, x, vars=None, svd=False):

        if svd:
            
            # リスト y に行列（パッチ行列 or fc入力）を追加していく
            y = []

            # --- stem (最初の conv1, bn1, relu) はパッチ取得しないでそのまま通す ---
            # out = self.conv1(x)
            # out = self.bn1(out)
            # out = F.relu(out)
            # print("out.shape: ", out.shape)

            fs, out = self.conv_to_linear(x, self.conv1)
            # conv1 の出力には bn1 と ReLU を適用
            out = self.bn1(out)
            out = F.relu(out)
            y.append(fs)
            # print("out.shape: ", out.shape)     # out.shape:  torch.Size([125, 64, 32, 32])
            # print("y[0].shape: ", y[0].shape)   # y[0].shape:  torch.Size([128000, 27])
            
            # --- layer1 ---
            # layer1 の block[0]
            out = self.svd_block_forward(self.layer1[0], out, y)
            # print("out.shape: ", out.shape)     # out.shape:  torch.Size([125, 64, 32, 32])
            # print("y[1].shape: ", y[1].shape)   # y[0].shape:  torch.Size([128000, 576])
            # print("y[2].shape: ", y[2].shape)   # y[2].shape:  torch.Size([128000, 576])

            # layer1 の block[1]
            out = self.svd_block_forward(self.layer1[1], out, y)
            # print("out.shape: ", out.shape)     # out.shape:  torch.Size([125, 64, 32, 32])
            # print("y[3].shape: ", y[3].shape)   # y[3].shape:  torch.Size([128000, 576])
            # print("y[4].shape: ", y[4].shape)   # y[4].shape:  torch.Size([128000, 576])

            # --- layer2 ---
            out = self.svd_block_forward(self.layer2[0], out, y)
            # print("out.shape: ", out.shape)     # out.shape:  torch.Size([125, 128, 16, 16])
            # print("y[5].shape: ", y[5].shape)   # y[5].shape:  torch.Size([32000, 576])
            # print("y[6].shape: ", y[6].shape)   # y[6].shape:  torch.Size([32000, 1152])
            # print("y[7].shape: ", y[7].shape)   # y[7].shape:  torch.Size([32000, 64])

            out = self.svd_block_forward(self.layer2[1], out, y)
            # print("out.shape: ", out.shape)     # out.shape:  torch.Size([125, 128, 16, 16])
            # print("y[8].shape: ", y[8].shape)   # y[8].shape:  torch.Size([32000, 1152])
            # print("y[9].shape: ", y[9].shape)   # y[9].shape:  torch.Size([32000, 1152])

            # --- layer3 ---
            out = self.svd_block_forward(self.layer3[0], out, y)
            # print("out.shape: ", out.shape)       # out.shape:  torch.Size([125, 256, 8, 8])
            # print("y[10].shape: ", y[10].shape)   # y[10].shape:  torch.Size([8000, 1152])
            # print("y[11].shape: ", y[11].shape)   # y[11].shape:  torch.Size([8000, 2304])
            # print("y[12].shape: ", y[12].shape)   # y[12].shape:  torch.Size([8000, 128])

            out = self.svd_block_forward(self.layer3[1], out, y)
            # print("out.shape: ", out.shape)       # out.shape:  torch.Size([125, 256, 8, 8])
            # print("y[13].shape: ", y[13].shape)   # y[13].shape:  torch.Size([8000, 2304])
            # print("y[14].shape: ", y[14].shape)   # y[14].shape:  torch.Size([8000, 2304])


            # --- layer4 ---
            out = self.svd_block_forward(self.layer4[0], out, y)
            # print("out.shape: ", out.shape)       # out.shape:  torch.Size([125, 512, 4, 4])
            # print("y[15].shape: ", y[15].shape)   # y[15].shape:  torch.Size([2000, 2304])
            # print("y[16].shape: ", y[16].shape)   # y[16].shape:  torch.Size([2000, 4608])
            # print("y[17].shape: ", y[17].shape)   # y[17].shape:  torch.Size([2000, 256])

            # 2025/04/17にコメントアウト
            # out = self.svd_block_forward(self.layer4[1], out, y)
            # # print("out.shape: ", out.shape)       # out.shape:  torch.Size([125, 512, 4, 4])
            # # print("y[18].shape: ", y[18].shape)   # y[18].shape:  torch.Size([2000, 4608])
            # # print("y[19].shape: ", y[19].shape)   # y[19].shape:  torch.Size([2000, 4608])

            # # avgpool->flatten->fc 入力もパッチにするなら下記など
            # out = self.avgpool(out)
            # # print("out.shape: ", out.shape)   # out.shape:  torch.Size([125, 512, 1, 1])
            # out = out.view(out.size(0), -1)
            # # print("out.shape: ", out.shape)   # out.shape:  torch.Size([125, 512])
            # y.append(out)

            print("len(y): ", len(y))
            # Layer 1 : (27, 51200)
            # Layer 2 : (576, 51200)
            # Layer 3 : (576, 51200)
            # Layer 4 : (576, 51200)
            # Layer 5 : (576, 51200)
            # Layer 6 : (576, 12800)
            # Layer 7 : (1152, 12800)
            # Layer 8 : (64, 12800)
            # Layer 9 : (1152, 12800)
            # Layer 10 : (1152, 64000)
            # Layer 11 : (1152, 16000)
            # Layer 12 : (2304, 16000)
            # Layer 13 : (128, 16000)
            # Layer 14 : (2304, 32000)
            # Layer 15 : (2304, 32000)
            # Layer 16 : (2304, 8000)
            # Layer 17 : (4608, 8000)
            # Layer 18 : (256, 8000)
            # Layer 19 : (4608, 8000)
            # Layer 20 : (4608, 8000)

            return y
        

        elif vars is not None:
            # 外部パラメータによる順伝搬
            idx = 0
            
            # conv1
            weight_conv1 = vars[idx]; idx += 1
            weight_bn1 = vars[idx]; idx += 1
            bias_bn1 = vars[idx]; idx += 1
            out = F.conv2d(x, weight_conv1, bias=None, stride=self.conv1.stride, padding=self.conv1.padding)
            out = F.batch_norm(out, self.bn1.running_mean, self.bn1.running_var,
                               weight=weight_bn1, bias=bias_bn1, training=True)
            out = F.relu(out)
            # print("out.shape: ", out.shape)
            
            # layer1
            out, idx = functional_layer_forward(out, self.layer1, vars, idx)

            # layer2
            out, idx = functional_layer_forward(out, self.layer2, vars, idx)
            
            # layer3
            out, idx = functional_layer_forward(out, self.layer3, vars, idx)
            
            # layer4
            out, idx = functional_layer_forward(out, self.layer4, vars, idx)
            
            # avgpoolとflatten
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            # print("out.shape: ", out.shape)
            
            # fc
            weight_fc = vars[idx]; idx += 1
            bias_fc = vars[idx]; idx += 1
            out_fc = F.linear(out, weight_fc, bias_fc)
            # print("out_fc.shape: ", out_fc.shape)

            return out_fc

        
        else:

            # print("x.shape: ", x.shape)   # x.shape:  torch.Size([32, 3, 32, 32])

            # 元のパラメータで順伝搬
            # conv1
            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)
            
            # layer1
            out = self.layer1(out)
            # print("out.shape: ", out.shape)  # out.shape:  torch.Size([32, 64, 32, 32])
            # assert False
            
            # layer2
            out = self.layer2(out)
            
            # layer3
            out = self.layer3(out)
            
            # layer4
            out = self.layer4(out)
            
            out = self.avgpool(out)
            out_flat = out.view(out.size(0), -1)
            # print("out_flat.shape: ", out_flat.shape)
            
            # 最終層
            out_fc = self.fc(out_flat)
            # print("out_fc.shape: ", out_fc.shape)
        
        return out_fc


    # リプレイバッファにデータを保存
    def push_to_mem(self, batch_x, batch_y, t, epoch):
        """
            Reservoir sampling to push subsampled stream of data points to replay buffer
            """
        if epoch > 1:
            return

        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()
        t = t.cpu()

        for i in range(batch_x.shape[0]):
            self.age += 1
            if len(self.M) < self.memories:
                self.M.append([batch_x[i], batch_y[i], t])
            else:
                p = np.random.randint(0, self.age)
                if p < self.memories:
                    self.M[p] = [batch_x[i], batch_y[i], t]
        
    
    
    def get_batch(self, x, y, t):
        """
            Given the new data points, create a batch of old + new data,
            where old data is sampled from the replay buffer
            """
        
        # 新しいデータ毎のtask idを作成
        t = (torch.ones(x.shape[0]).int() * t)

        if len(self.M) > 0:
            MEM = self.M
            order = np.arange(len(MEM))
            np.random.shuffle(order)
            index = order[:min(x.shape[0], len(MEM))]

            x = x.cpu()
            y = y.cpu()

            for k, idx in enumerate(index):
                ox, oy, ot = MEM[idx]
                x = torch.cat((x, ox.unsqueeze(0)), 0)
                y = torch.cat((y, oy.unsqueeze(0)), 0)
                t = torch.cat((t, ot.unsqueeze(0)), 0)

        # handle gpus if specified
        if self.cuda:
            x = x.cuda()
            y = y.cuda()
            t = t.cuda()

        return x, y, t


    def zero_grads(self):
        
        self.zero_grad()

        if len(self.M_vec) > 0 and self.opt.fsdgpm_method in ['dgpm', 'xdgpm']:
            self.lambdas.zero_grad()


    def meta_loss(self, x, y, tasks, fast_weights=None):
        """
            Get loss of multiple tasks tasks
            """
        outputs = self.net_forward(x, fast_weights)
        
        loss = 0.0

        for task in np.unique(tasks.data.cpu().numpy()):
            task = int(task)
            idx = torch.nonzero(tasks == task).view(-1)

            loss += self.loss_ce(outputs[idx], y[idx]) * len(idx)



        return loss/len(y)



    def take_loss(self, x, y, t, fast_weights=None, criterion=None):

        outputs = self.net_forward(x, fast_weights)
        loss = criterion(outputs, y)

        return loss



    def update_weight(self, x, y, t, fast_weights, criterion):

        loss = self.take_loss(x, y, t, fast_weights, criterion)
        if fast_weights is None:
            fast_weights = self.get_params()
        
        # print("fast_weights: ", fast_weights)

        # NOTE if we want higher order grads to be allowed, change create_graph=False to True
        # print("self.second_order: ", self.second_order)
        graph_required = self.second_order
        grads = list(torch.autograd.grad(loss, fast_weights, create_graph=graph_required, retain_graph=graph_required,
                                         allow_unused=True))

        # 勾配の射影
        if len(self.M_vec) > 0:
            grads = self.grad_projection(grads)
            # print("len(grads): ", len(grads))
            # a = 1

        # 勾配のクリッピング
        for i in range(len(grads)):
            if grads[i] is not None:
                grads[i] = torch.clamp(grads[i], min=-self.opt.grad_clip_norm, max=self.opt.grad_clip_norm)


        # 重みへの摂動を適用
        if self.opt.sharpness:
            fast_weights = list(
                map(lambda p: p[1] + p[0] * self.eta1 if p[0] is not None else p[1], zip(grads, fast_weights)))
        else:
            fast_weights = list(
                map(lambda p: p[1] - p[0] * self.eta1 if p[0] is not None else p[1], zip(grads, fast_weights)))

        return fast_weights
    

    def define_lambda_params(self):
        assert len(self.M_vec) > 0

        # Setup learning parameters
        self.lambdas = nn.ParameterList([])
        for i in range(len(self.M_vec)):
            self.lambdas.append(nn.Parameter(self.opt.lam_init * torch.ones((self.M_vec[i].shape[1]), requires_grad=True)))

        if self.cuda:
            self.lambdas = self.lambdas.cuda()

        return


    def update_opt_lambda(self, lr=None):
        if lr is None:
            lr = self.eta2
        self.opt_lamdas = torch.optim.SGD(list(self.lambdas.parameters()), lr=lr, momentum=self.opt.momentum)

        return



    # def grad_projection(self, grads):
    #     """
    #         get the projection of grads on the subspace spanned by GPM
    #         """
    #     j = 0

    
    # def grad_projection(self, grads):
    #     """
    #         get the projection of grads on the subspace spanned by GPM
    #         """
    #     j = 0
    #     ndim_1 = 0
    #     ndim_2 = 0
    #     ndim_4 = 0
    #     for i in range(len(grads)):
    #         g = grads[i]
    #         # print("g.ndim: ", g.ndim)
    #         if g is not None:
    #             if g.ndim == 1:
    #                 ndim_1 += 1
    #             elif g.ndim == 2:
    #                 ndim_2 += 1
    #             elif g.ndim == 4:
    #                 ndim_4 += 1
    #             else:
    #                 assert False
    #     print("ndim_1: ", ndim_1)  # ndim_1:  41
    #     print("ndim_2: ", ndim_2)  # ndim_2:  1
    #     print("ndim_4: ", ndim_4)  # ndim_4:  20


    #     assert False

    def grad_projection(self, grads):
        """
            get the projection of grads on the subspace spanned by GPM
            """
        j = 0
        # print("self.lambdas: ", self.lambdas)
        # print("len(self.lambdas): ", len(self.lambdas))
        # print("len(grads): ", len(grads))
        # print("len(self.M_vec): ", len(self.M_vec))
        for i in range(len(grads)):
            # only update conv weight and fc weight
            # ignore perturbations with 1 dimension (e.g. BN, bias)
            if grads[i] is None:
                continue
            if grads[i].ndim <= 1:
                continue
            # print("grads[i].ndim: ", grads[i].ndim)
            if j < len(self.M_vec):
                if self.opt.fsdgpm_method in ['dgpm', 'xdgpm']:
                    # lambdas = torch.sigmoid(self.args.tmp * self.lambdas[j]).reshape(-1)
                    lambdas = self.lambdas[j]
                else:
                    lambdas = torch.ones(self.M_vec[j].shape[1])

                if self.cuda:
                    self.M_vec[j] = self.M_vec[j].cuda()
                    lambdas = lambdas.cuda()

                if grads[i].ndim == 4:
                    # rep[i]: n_samples * n_features
                    grad = grads[i].reshape(grads[i].shape[0], -1)
                    grad = torch.mm(torch.mm(torch.mm(grad, self.M_vec[j]), torch.diag(lambdas)), self.M_vec[j].T)
                    grads[i] = grad.reshape(grads[i].shape).clone()
                else:
                    grads[i] = torch.mm(torch.mm(torch.mm(grads[i], self.M_vec[j]), torch.diag(lambdas)), self.M_vec[j].T)

                j += 1
        
        # print("len(grads): ", len(grads))

        return grads
    
    
    
    def train_restgpm(self):
        """
            update grad to the projection on the rest of subspace spanned by GPM
            """
        j = 0

        for p in self.parameters():
            # only handle conv weight and fc weight
            if p.grad is None:
                continue
            if p.grad.ndim != 2 and p.grad.ndim != 4:
                continue
            if j < len(self.M_vec):
                if self.opt.method in ['fs-dgpm'] and self.opt.fsdgpm_method in ['dgpm', 'xdgpm']:
                    # lambdas = torch.sigmoid(self.args.tmp * self.lambdas[j]).reshape(-1)
                    lambdas = self.lambdas[j]
                else:
                    lambdas = torch.ones(self.M_vec[j].shape[1])

                if self.cuda:
                    self.M_vec[j] = self.M_vec[j].cuda()
                    lambdas = lambdas.cuda()

                if p.grad.ndim == 4:
                    # rep[i]: n_samples * n_features
                    grad = p.grad.reshape(p.grad.shape[0], -1)
                    grad -= torch.mm(torch.mm(torch.mm(grad, self.M_vec[j]), torch.diag(lambdas)), self.M_vec[j].T)
                    p.grad = grad.reshape(p.grad.shape).clone()
                else:
                    p.grad -= torch.mm(torch.mm(torch.mm(p.grad, self.M_vec[j]), torch.diag(lambdas)), self.M_vec[j].T)

                j += 1


    def set_gpm_by_svd(self, threshold):
        """
            Get the bases matrix (GPM) based on data sampled from replay buffer
            """
        assert len(self.M) > 0

        self.M_vec = []
        self.M_val = []

        # リプレイバッファからサンプル抽出
        index = np.arange(len(self.M))
        np.random.shuffle(index)

        # 
        for k, idx in enumerate(index):
            if k < min(self.mem_batch_size, len(self.M)):
                ox, oy, ot = self.M[idx]
                if k == 0:
                    mx = ox.unsqueeze(0)
                else:
                    mx = torch.cat((mx, ox.unsqueeze(0)), 0)

        # print("mx.shape: ", mx.shape)   # mx.shape:  torch.Size([64, 3, 32, 32])

        if self.cuda:
            mx = mx.cuda()

        # 中間層の出力と最終出力を獲得する
        with torch.no_grad():
            rep = self.net_forward(mx, svd=True)
            # print("rep[1].shape: ", rep[1].shape)  # rep[1].shape:  torch.Size([64, 100])
        
        # 
        for i in range(len(rep)):
            # rep[i]: n_samples * n_features
            assert rep[i].ndim == 2

            # SVD
            u, s, v = torch.svd(rep[i].detach().cpu())
            print()
            print("u.shape: ", u.shape)
            print("s.shape: ", s.shape)
            print("v.shape: ", v.shape)

            if threshold[i] < 1:
                r = torch.cumsum(s ** 2, 0) / torch.cumsum(s ** 2, 0)[-1]
                k = torch.searchsorted(r, threshold[i]) + 1

            elif threshold[i] > 10:
                # differ with thres = 1
                k = int(threshold[i])
            else:
                k = len(s)

            if k > 0:
                self.M_vec.append(v[:, :k].cpu())
                self.M_val.append(s[:k].cpu())
    

    

def ResNet18(nf=32, nclass=100, seed=777, opt=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], nf, nclass, seed, opt=opt)








# # GPMの公式実装に準拠したバージョン
# # 可能な限り全てのモデル構造を統一したいので，書き換える
# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(in_planes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
#                           stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
#             )
#         self.act = OrderedDict()
#         self.count = 0

#     def forward(self, x):
#         self.count = self.count % 2 
#         self.act['conv_{}'.format(self.count)] = x
#         self.count +=1
#         out = relu(self.bn1(self.conv1(x)))
#         self.count = self.count % 2 
#         self.act['conv_{}'.format(self.count)] = out
#         self.count +=1
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = relu(out)
#         return out

# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, nf, nclass):
#         super(ResNet, self).__init__()

#         self.in_planes = nf
#         self.conv1 = conv3x3(3, nf * 1, 1)
#         self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
#         self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
#         # self.linear=torch.nn.ModuleList()
#         self.fc = nn.Linear(nf * 8 * block.expansion * 4, nclass, bias=False)
#         self.max_num_classes = nclass
#         self.act = OrderedDict()

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         bsz = x.size(0)
#         self.act['conv_in'] = x.view(bsz, 3, 32, 32)
#         out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32)))) 
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = avg_pool2d(out, 2)
#         out = out.view(out.size(0), -1)
        
#         return self.fc(out)
#         # y=[]
#         # for t,i in self.taskcla:
#         #     y.append(self.linear[t](out))
#         # return y