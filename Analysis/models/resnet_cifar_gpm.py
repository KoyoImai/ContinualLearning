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
    def __init__(self, block, num_blocks, nf=64, nclass=100, seed=777, zero_init_residual=False):
        super(ResNet, self).__init__()
        
        seed_everything(seed=seed)

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
        self.act['conv_in'] = x.view(bsz, 3, 32, 32)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32)))) 
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

    

def ResNet18(nf=32, nclass=100, seed=777):
    return ResNet(BasicBlock, [2, 2, 2, 2], nf, nclass, seed)








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