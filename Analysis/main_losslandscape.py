
import os

import argparse
import numpy as np

import torch



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # 手法の決定
    parser.add_argument("--method", type=str, default="cclis")

    # データセットの決定
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--task_list", required=True, nargs="*", type=int)

    # 事前学習済みモデルのパス
    parser.add_argument("--pretrained_path", type=str, default="./")
    
    # 手法毎のハイパラ（co2l，cclis）
    parser.add_argument("--temp", type=float, default=0.1)

    # 手法毎のハイパラ（cclis）
    parser.add_argument('--wo_is', default=False, action='store_true')




    # その他
    parser.add_argument("--seed", type=int, default=777)

    opt = parser.parse_args()


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
    # （プログラムを追加）


    # モデル，損失関数，データローダーを渡して誤差局面を計算・可視化
    # （プログラムを追加）




if __name__ == "__main__":
    main()



