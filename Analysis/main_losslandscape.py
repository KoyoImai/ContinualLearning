
import os
import random

import argparse
import numpy as np

import torch


from util import seed_everything

from tools_losslandscape.make_dataloaders import make_dataloader
from tools_losslandscape.landscape import plot_loss_landscape_2d




def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # 手法の決定
    parser.add_argument("--method", type=str, default="cclis")

    # データセットの決定
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument('--data_folder', type=str, default='/home/kouyou/Datasets/', help='path to custom dataset')
    parser.add_argument("--cls_list", required=True, nargs="*", type=int)

    # 事前学習済みモデルのパス
    parser.add_argument("--pretrained_path", type=str, default="./")
    
    # 手法毎のハイパラ（co2l，cclis）
    parser.add_argument("--temp", type=float, default=0.1)

    # 手法毎のハイパラ（cclis）
    parser.add_argument('--wo_is', default=False, action='store_true')

    # 手法毎のハイパラ
    parser.add_argument('--not_asym', default=False, action='store_true')

    # ハイパラ
    parser.add_argument('--batch_size', type=int, default=512)


    # ---- runtime / system ----
    parser.add_argument('--cuda', action='store_true', help='use CUDA if available')
    parser.add_argument('--threads', type=int, default=2, help='num dataloader workers')
    parser.add_argument('--ngpu', type=int, default=1, help='num GPUs per process')

    # ---- landscape ranges ----
    parser.add_argument('--x', type=str, default='-1:1:51', help='xmin:xmax:xnum')
    parser.add_argument('--y', type=str, default=None, help='ymin:ymax:ynum (2D when set)')

    # ---- direction generation ----
    parser.add_argument('--dir_type', type=str, default='random', choices=['random','states'],
                        help='random directions or states-difference for x-axis')
    parser.add_argument('--base_ckpt', type=str, default=None, help='path to base checkpoint (theta*)')
    parser.add_argument('--second_ckpt', type=str, default=None, help='path to second checkpoint (for states direction)')
    parser.add_argument('--skip_bn_bias', action='store_true', help='ignore BN/bias in direction')
    parser.add_argument('--norm', type=str, default='filter', choices=['filter','layer','weight','none'],
                        help='direction normalization (filter recommended)')

    # ---- evaluation control ----
    parser.add_argument('--max_batches', type=int, default=None, help='avg over at most K batches per grid point')
    parser.add_argument('--save_png', type=str, default='landscape_2d.png', help='output figure path')

    # その他
    parser.add_argument("--seed", type=int, default=777)

    opt = parser.parse_args()


    # parse x/y ranges into numbers
    try:
        opt.xmin, opt.xmax, opt.xnum = [float(a) for a in opt.x.split(':')]
        if opt.y:
            opt.ymin, opt.ymax, opt.ynum = [float(a) for a in opt.y.split(':')]
        else:
            opt.ymin = opt.ymax = opt.ynum = None
    except Exception as e:
        raise ValueError(f'Bad format for --x/--y. Use like "-1:1:51". ({e})')


    return opt



def preparation(opt):

    # データセット毎にタスク数・タスク毎のクラス数を決定
    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.size = 32
    if opt.dataset == 'cifar100':
        opt.n_cls = 100
        opt.cls_per_task = 5
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    else:
        pass



# モデルと損失関数の設計
def make_setup(opt):

    if opt.method == "co2l":

        from losses.loss_co2l import SupConLoss

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_co2l import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed)
        criterion = SupConLoss(temperature=opt.temp, not_asym=opt.not_asym)
        
    
    elif opt.method == "cclis":

        from losses.loss_cclis import ISSupConLoss

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_cclis import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        criterion = ISSupConLoss(temperature=opt.temp, opt=opt)
        
    
    elif opt.method == "er":

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_er import BackboneResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = BackboneResNet(name='resnet18', head='linear', feat_dim=opt.n_cls, seed=opt.seed)
        criterion = torch.nn.CrossEntropyLoss()
    

    return model, criterion





def main():

    # コマンドライン引数の処理
    opt = parse_option()


    # データローダ作成の前処理
    preparation(opt)


    # seed値の固定
    seed_everything(opt.seed)


    # modelと損失関数の用意
    # モデルはresnet18，損失関数は交差エントロピー・対照損失（・蒸留損失など）が主な対象
    model, criterion = make_setup(opt=opt)


    # 事前学習済みモデルの読み込み
    ckpt = torch.load(opt.pretrained_path, map_location='cpu')
    state_dict = ckpt['model']
    if isinstance(model, torch.nn.DataParallel):
        new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)


    # データローダーの作成
    dataloaders = make_dataloader(opt)


    # モデルと損失関数をgpu上に配置
    if torch.cuda.is_available():
        model = model.cuda().eval()
        criterion = criterion.cuda()

    # データローダーを取り出し
    train_laoder, val_loader = dataloaders

    # 可視化範囲
    x_range = (opt.xmin, opt.xmax, int(opt.xnum))
    if opt.ymin is not None:
        y_range = (opt.ymin, opt.ymax, int(opt.ynum))
    else:
        # y 未指定の場合は 1D プロット相当（y=0 の1ライン）として 2D の薄い格子にする
        y_range = (0.0, 0.0, 1)
    

    # 正規化フラグ：公式実装準拠なら filter を推奨
    filter_normalize = (opt.norm == 'filter')

    # モデル，損失関数，データローダーを渡して誤差局面を計算・可視化
    plot_loss_landscape_2d(model=model, criterion=criterion, loader=dataloaders)



if __name__ == "__main__":
    main()



