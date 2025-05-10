import os
import random
import argparse
import numpy as np
import logging


from scipy import optimize
import torch
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset
import torch.optim.lr_scheduler as lr_scheduler




def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # 手法
    parser.add_argument('--method', type=str, default="",
                        choices=['er', 'co2l', 'gpm', 'lucir', 'fs-dgpm', 'cclis', 'supcon', 'supcon-joint', 'simclr',
                                 'cclis-wo', 'cclis-wo-ss', 'cclis-wo-is'])

    # logの名前（実行毎に変えてね）
    parser.add_argument('--log_name', type=str, default="practice")


    # データセット周り
    parser.add_argument('--data_folder', type=str, default='/home/kouyou/Datasets/', help='path to custom dataset')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet', 'path'], help='dataset')


    # 最適化手法
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    
    # 学習条件
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--vanilla_batch_size', type=int, default=500)


    # classifierの学習条件(Co2Lなど線形分類で後から評価する手法用)
    parser.add_argument('--linear_epochs', type=int, default=100)
    parser.add_argument('--linear_lr', type=float, default=0.1)
    parser.add_argument('--linear_momentum', type=float, default=0.9)
    parser.add_argument('--linear_weight_decay', type=float, default=0)
    parser.add_argument('--linear_batch_size', type=int, default=256)

    # 継続学習的設定
    parser.add_argument('--mem_size', type=int, default=500)
    parser.add_argument('--mem_type', type=str, default="ring",
                        choices=["reservoir", "ring", "herding"])
    
    # 使用するモデル
    parser.add_argument("--model", type=str, default="resnet18")

    # 手法毎のハイパラ（共通）
    parser.add_argument("--temp", type=float, default=2)
    parser.add_argument("--lamda", type=float, default=5)

    # 手法毎のハイパラ（co2l）
    parser.add_argument('--current_temp', type=float, default=0.2)
    parser.add_argument('--past_temp', type=float, default=0.1)
    parser.add_argument('--distill_power', type=float, default=0.1)

    # 手法毎のハイパラ（gpm & fs-dgpm）
    parser.add_argument('--threshold', type=float, default=0.965)

    # 手法毎のハイパラ（lucir）
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--dist", type=float, default=0.5)
    parser.add_argument("--lw_mr", type=float, default=1)

    # 手法毎のハイパラ（cclis）
    parser.add_argument('--distill_type', type=str, default="PRD")
    parser.add_argument('--max_iter', type=int, default=5,
                        help='iterations of the score computing')
    parser.add_argument('--learning_rate_prototypes', type=float, default=0.01)
    parser.add_argument('--cosine', default=False, action='store_true')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')

    # 手法毎のハイパラ（fs-dgpm）
    # parser.add_argument('--inner_batches', type=int, default=2)
    parser.add_argument('--inner_steps', type=int, default=2)
    parser.add_argument('--freeze_bn', default=False, action='store_true')
    parser.add_argument('--second_order', default=False, action='store_true', help='')
    parser.add_argument('--mem_batch_size', type=int, default=64)
    parser.add_argument('--grad_clip_norm', type=float, default=2.0, help='Clip the gradients by this value')
    parser.add_argument('--sharpness', default=False, action='store_true', help='')
    parser.add_argument('--eta1', type=float, default=1e-2, help='update step size of weight perturbation')
    parser.add_argument('--eta2', type=float, default=1e-2, help='learning rate of lambda(soft weight for basis)')
    parser.add_argument('--fsdgpm_method', type=str, default='xdgpm', choices=['xdgpm', 'dgpm', 'xgpm'])
    parser.add_argument('--thres_add', type=float, default=0.003, help='thres_add')
    parser.add_argument('--thres_last', type=float, default=0.99999999999, help='thres_last')
    parser.add_argument('--lam_init', type=float, default=1.0, help='temperature for sigmoid')
    parser.add_argument('--tmp', type=float, default=10, help='temperature for sigmoid')


    # その他の条件
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--date', type=str, default="2001_05_02")
    parser.add_argument('--earlystop', default=False, action='store_true', help='')


    opt = parser.parse_args()

    return opt