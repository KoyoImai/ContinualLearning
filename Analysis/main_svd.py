import os

import argparse
import numpy as np

import torch


from dataloaders import set_loader
from extract import extract_features
from analysis_tools.analysis_svd import svd, svd_all_layers

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # 評価対象関連
    parser.add_argument("--method", type=str, default="cclis")
    parser.add_argument("--dataset", type=str, default="cifar100")

    # データセットのディレクトリ
    parser.add_argument('--data_folder', type=str, default='/home/kouyou/Datasets/', help='path to custom dataset')
    
    # modelとlogまでのパス
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--log_path", type=str, default=None)

    # Projectorの使用
    parser.add_argument("--projector", default=False, action='store_true')

    # 各層の出力を分析対象にするかどうかの指定など
    parser.add_argument('--block_type', type=str, default="block",
                        choices=["block", "basicblock", "conv"])
    parser.add_argument('--flatten_type', type=str, default="flatten",
                        choices=["flatten", "avgpool"])

    # その他
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_dp", default=False, action='store_true')

    opt = parser.parse_args()

    # データセット毎にタスク数・タスク毎のクラス数を決定
    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.size = 32
    if opt.dataset == 'cifar100':
        opt.n_cls = 100
        opt.cls_per_task = 10
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    else:
        pass

    # 総タスク数
    opt.n_task = opt.n_cls // opt.cls_per_task

    opt.log_dir = os.path.dirname(opt.log_path)
    opt.log_name = os.path.basename(opt.log_dir)
    # print("opt.log_dir: ", opt.log_dir)
    # print("opt.log_name: ", opt.log_name)

    # 可視化結果や表の保存先
    # opt.save_path = f'{opt.log_dir}/annalyze/svd'
    opt.save_path = f'./logs/{opt.method}/{opt.log_name}/annalyze/svd'


    # ディレクトリ作成
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)



    return opt




def make_setup(opt):

    if opt.method == "co2l":

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_co2l import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed)
    
    elif opt.method == "cclis":

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_cclis import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
    
    elif opt.method == "er":

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_er import BackboneResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = BackboneResNet(name='resnet18', head='linear', feat_dim=opt.n_cls, seed=opt.seed)

    elif opt.method in ["supcon", "supcon-joint"]:

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_co2l import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed)

    else:

        assert False
    

    if torch.cuda.is_available():
        
        print(torch.cuda.device_count())
        if opt.use_dp:
            model = torch.nn.DataParallel(model)

        model = model.cuda()

    return model




def main():

    # コマンドライン引数の処理
    opt = parse_option()

    # modelの作成
    model = make_setup(opt=opt)
    print("model: ", model)

    # replay_indicesの初期化
    replay_indices = None

    # rank保存
    all_rank = {}
    alldata_task_rank = {}
    alldata_class_rank = {}

    # 各タスクのデータを用いて分析（SVD）
    for target_task in range(0, opt.n_task):

        # 現在タスクの更新
        opt.target_task = target_task

        # モデルパラメータの読み込み
        # ckpt_path = f"{opt.model_path}/model_{opt.target_task:02d}.pth"
        ckpt_path = f"{opt.model_path}/model_00.pth"
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['model']
        # model.load_state_dict(state_dict)

        if isinstance(model, torch.nn.DataParallel):
            new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

        # 場合によってはここをコメントアウト
        # if opt.use_dp:
        #     model = torch.nn.DataParallel(model)
        #     print("model; ", model)

        # リプレイサンプルいの読み込み
        if opt.target_task == 0:
            replay_indices = np.array([])
        else:
            file_path = f"{opt.log_path}/replay_indices_{opt.target_task}.npy"
            # file_path = f"{opt.log_path}/replay_indices_0.npy"
            replay_indices = np.load(file_path)

        # データローダーの作成（バッファ内のデータも含めて）
        data_loaders  = set_loader(opt, replay_indices)


        # 特徴量を抽出（これまでのタスクに含まれるすべてのサンプル）
        features, labels, layer_outputs = extract_features(opt=opt, model=model, data_loader=data_loaders["train"])
        print("1 features.shape: ", features.shape)
        print("1 labels.shape: ", labels.shape)
        print()

        # SVDによる分析（タスク，クラス毎に特徴ベクトルを分割）
        results = svd(opt=opt, features=features, labels=labels, cls_per_task=opt.cls_per_task, name="alldata")  # タスク毎にSVD
        results = svd(opt=opt, features=features, labels=labels, name="alldata")                                 # クラス毎にSVD

        # SVDによる分析（すべての特徴ベクトルを一度にSVD）
        results = svd(opt=opt, features=features, labels=labels, name="alldata", mode="all")  # タスク毎にSVD

        # 各層の出力毎にSVD
        # results = svd_all_layers(opt=opt, layer_outputs=layer_outputs, labels=labels, name="alldata", mode="cls")



        # 特徴量を抽出（現在のタスクに含まれるサンプルとリプレイサンプル）
        features, labels, layer_outputs = extract_features(opt=opt, model=model, data_loader=data_loaders["trainv2"])
        print("2 features.shape: ", features.shape)
        print("2 labels.shape: ", labels.shape)
        
        # SVDによる分析
        results = svd(opt=opt, features=features, labels=labels, cls_per_task=opt.cls_per_task, name="replay")  # タスク毎にSVD
        results = svd(opt=opt, features=features, labels=labels, name="replay")                                 # クラス毎にSVD








if __name__ == "__main__":
    main()