
import os
import copy
import argparse
import logging
import numpy as np


import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


from dataloaders import set_buffer, set_loader, set_loader_eval, set_loader_eval4timnet
from preprocesses import pre_process
from postprocesses import post_process
from train import train, eval, eval4timnet


from util import seed_everything, save_model





def parse_option():
    
    parser = argparse.ArgumentParser('argument for training')

    # 基本的な実験設定
    parser.add_argument("--log_name", type=str, default="test")
    parser.add_argument('--method', type=str, default='cclis', choices=['er', 'co2l', 'cclis', 'prco', 'prco-fimcl', 'prco-fimclv2', 'prco-fimclv3', 'prco-efm'])

    # データセット関連
    parser.add_argument('--data_folder', type=str, default='/home/kouyou/Datasets/', help='path to custom dataset')
    parser.add_argument('--data_order', type=str, default="original")
    parser.add_argument('--dataset', type=str, default="cifar100", choices=["cifar10", "cifar100", "tiny-imagenet", "imagenet"])


    # 最適化設定
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--start_epoch', type=int, default=500)
    # parser.add_argument('--batch_size', type=int, default=512)
    # parser.add_argument('--learning_rate', type=float, default=0.03)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--feat_dim', type=int, default=128)


    # 線形分類層の最適化設定
    parser.add_argument('--linear_batch_size', type=int, default=256)
    parser.add_argument('--linear_learning_rate', type=float, default=1.0)
    parser.add_argument('--linear_epochs', type=int, default=10)
    parser.add_argument('--linear_momentum', type=float, default=0.9)
    parser.add_argument('--linear_weight_decay', type=float, default=0)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')

    # 継続学習的設定
    parser.add_argument("--mem_type", type=str, default="ring")
    parser.add_argument('--mem_size', type=int, default=500)
    parser.add_argument('--offline', action='store_true', help='offline learning')

    # co2lのハイパーパラメータ
    # parser.add_argument('--temp_co2l', type=float, default=0.5)
    # parser.add_argument('--current_temp', type=float, default=0.2)
    # parser.add_argument('--past_temp', type=float, default=0.1)
    # parser.add_argument('--distill_power', type=float, default=0.1)
    # parser.add_argument('--not_asym', default=False, action='store_true')

    # cclisのハイパーパラメータ
    # parser.add_argument('--temp_cclis', type=float, default=0.5)
    # parser.add_argument('--wo_is', action='store_true')
    # parser.add_argument('--learning_rate_prototypes', type=float, default=0.01)
    # parser.add_argument('--cosine', default=False, action='store_true')
    # parser.add_argument('--distill_type', type=str, default="PRD")
    # parser.add_argument('--max_iter', type=int, default=5, help='iterations of the score computing')

    # prco
    

    # その他の設定
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--date', type=str, default='2024_0101')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=50)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--epoch_save', default=False, action='store_true')   # エポック毎にモデルを保存


    # 評価関係
    parser.add_argument("--target_task", type=int, default=None)
    parser.add_argument("--target_epoch", type=int, default=None)


    # 

    opt = parser.parse_args()

    return opt


def preparation(opt):

    # データセット毎にタスク数・タスク毎のクラス数を決定
    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        if opt.offline:
            opt.cls_per_task = 10
        else:
            opt.cls_per_task = 2
        opt.size = 32
    if opt.dataset == 'cifar100':
        opt.n_cls = 100
        if opt.offline:
            opt.cls_per_task = 100
        else:
            opt.cls_per_task = 20
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        if opt.offline:
            opt.cls_per_task = 200
        else:
            opt.cls_per_task = 20
        opt.size = 64
    else:
        pass

    # 総タスク数
    opt.n_task = opt.n_cls // opt.cls_per_task

    # モデルの保存，実験記録などの保存先パス
    if opt.data_folder is None:
        opt.data_folder = '~/data/'
    opt.model_path = f'./logs/{opt.method}/{opt.log_name}/model/'      # modelの保存先
    opt.explog_path = f'./logs/{opt.method}/{opt.log_name}/exp_log/'   # 実験記録の保存先
    opt.mem_path = f'./logs/{opt.method}/{opt.log_name}/mem_log/'      # リプレイバッファ内の保存先
    opt.result_path = f'./logs/{opt.method}/{opt.log_name}/result/'    # 結果の保存先

    # ディレクトリ作成
    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.isdir(opt.explog_path):
        os.makedirs(opt.explog_path)
    if not os.path.isdir(opt.mem_path):
        os.makedirs(opt.mem_path)
    if not os.path.isdir(opt.result_path):
        os.makedirs(opt.result_path)





def make_setup(opt):

    if opt.method in ["er"]:

        assert False

    elif opt.method in ["co2l"]:

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_co2l import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=opt.feat_dim, seed=opt.seed)


    elif opt.method in ["cclis"]:

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_cclis import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False

        model = SupConResNet(name='resnet18', head='mlp', feat_dim=opt.feat_dim, seed=opt.seed, opt=opt)

    elif opt.method in ["prco", 'prco-efm']:

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_prco import SupConResNet
        elif opt.dataset in ["imagenet"]:
            assert False

        model = SupConResNet(name='resnet18', head='mlp', feat_dim=opt.feat_dim, seed=opt.seed, opt=opt)
    
    elif opt.method in ["prco-fimcl", "prco-fimclv2"]:

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_prco_fimcl import SupConResNet
        elif opt.dataset in ["imagenet"]:
            assert False

        model = SupConResNet(name='resnet18', head='mlp', feat_dim=opt.feat_dim, seed=opt.seed, opt=opt)

    else:

        assert False
    

    if torch.cuda.is_available():
        model = model.cuda()

        model = torch.nn.DataParallel(model)


    return model, None
    


def main():

    # コマンドライン引数の処理
    opt = parse_option()

    # 乱数のシード固定（既存のコードに追加）
    seed_everything(opt.seed)

    # logの名前
    opt.log_name = f"{opt.log_name}_{opt.method}_{opt.mem_type}{opt.mem_size}_{opt.dataset}_seed{opt.seed}_date{opt.date}"
    print("log_name: ", opt.log_name)

    # データローダ作成の前処理
    preparation(opt)


    # modelの作成，損失関数の作成，Optimizerの作成
    model, method_tools = make_setup(opt)
    # print("model: ", model)

    # パラメータの読み込み
    ckpt_path = f"{opt.model_path}/task{opt.target_task:02d}/model_epoch{opt.target_epoch:03d}.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model']
    model.load_state_dict(state_dict)



    # リプレイサンプルの読み込み
    if opt.target_task == 0:
        replay_indices = np.array([])
    else:
        file_path = f"{opt.mem_path}/replay_indices_{opt.target_task}.npy"
        # file_path = f"{opt.log_path}/replay_indices_0.npy"
        replay_indices = np.load(file_path)
    # print("replay_indices.shape: ", replay_indices.shape)

    # データローダーの作成（バッファ内のデータも含めて）
    dataloader = set_loader_eval4timnet(opt, model, replay_indices, method_tools)

    eval4timnet(model=model, dataloader=dataloader, opt=opt)







if __name__ == "__main__":
    main()

