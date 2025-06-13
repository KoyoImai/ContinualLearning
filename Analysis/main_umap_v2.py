"""
タスク毎に特徴量を全て取り出して一括でumapモデルをfitする
"""


import os

import argparse
import numpy as np

import torch

from dataloaders import set_loader
from extract import extract_features
from analysis_tools.analysis_score import Eval_cluster

from util import write_csv_analysis

import seaborn as sns
import matplotlib.pyplot as plt
import umap

import itertools


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
    parser.add_argument("--start_epoch", type=int, default=500)
    parser.add_argument("--epoch", type=int, default=100)

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

    parser.add_argument('--offline', default=False, action='store_true')

    opt = parser.parse_args()


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
            opt.cls_per_task = 10
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

    opt.log_dir = os.path.dirname(opt.log_path)
    opt.log_name = os.path.basename(opt.log_dir)
    # print("opt.log_dir: ", opt.log_dir)
    # print("opt.log_name: ", opt.log_name)

    # 可視化結果や表の保存先
    # opt.save_path = f'{opt.log_dir}/annalyze/svd'
    opt.save_path = f'./logs/{opt.method}/{opt.log_name}/annalyze/score/offline_{opt.offline}'


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
            # model = torch.nn.DataParallel(model)
            model.encoder = torch.nn.DataParallel(model.encoder)

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

    # 全タスク・全特徴の蓄積
    all_features = []
    all_labels = []
    all_task_ids = []

    # 各タスクのデータを用いて分析（UMAP）
    for target_task in range(0, opt.n_task):

        # 現在タスクの更新
        opt.target_task = target_task

        # モデルパラメータの読み込み
        ckpt_path = f"{opt.model_path}/model_{opt.target_task:02d}.pth"
        # ckpt_path = f"{opt.model_path}/model_00.pth"
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['model']
        # model.load_state_dict(state_dict)


        if isinstance(model.encoder, torch.nn.DataParallel):
            # ② キーごとに変換。たとえば "encoder.conv1.weight" → "encoder.module.conv1.weight"
            new_dict = {}
            for k, v in state_dict.items():
                if k.startswith("encoder."):
                    # "encoder." のあとに来る部分を取り出し、
                    # 先頭に "encoder.module." を付与する
                    suffix = k[len("encoder."):]              # 例: "conv1.weight"
                    new_key = "encoder.module." + suffix       # → "encoder.module.conv1.weight"
                else:
                    # head.1.weight などは変えずにそのまま載せる
                    new_key = k
                new_dict[new_key] = v

            model.load_state_dict(new_dict)

        else:
            # model.encoder が DataParallel でなければ，
            # 保存時と同じキー構造 ("encoder.conv1.weight" ...) のまま読み込む
            model.load_state_dict(state_dict)


        # リプレイサンプルいの読み込み
        if opt.target_task == 0:
            replay_indices = np.array([])
        else:
            file_path = f"{opt.log_path}/replay_indices_{opt.target_task}.npy"
            # file_path = f"{opt.log_path}/replay_indices_0.npy"
            replay_indices = np.load(file_path)

        # データローダーの作成（バッファ内のデータも含めて）
        data_loaders  = set_loader(opt, replay_indices)

        # # 訓練用データを使用
        # features, labels, layer_outputs = extract_features(opt=opt, model=model, data_loader=data_loaders["train"])
        # print("1 features.shape: ", features.shape)
        # print("1 labels.shape: ", labels.shape)

        # 検証用データを使用
        features, labels, layer_outputs = extract_features(opt=opt, model=model, data_loader=data_loaders["val"])
        print("1 features.shape: ", features.shape)
        print("1 labels.shape: ", labels.shape)


        # 特徴量とラベルをnumpy配列に変換
        if torch.is_tensor(features):
            features = features.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()


        # 特徴量，ラベル，taskidを格納
        all_features.append(features)
        all_labels.append(labels)

    

    # ---------- 一括でUMAP学習 ----------
    all_features_np = np.concatenate(all_features, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    print("all_features_np.shape: ", all_features_np.shape)   # all_features_np.shape:  (30000, 512)
    print("all_labels_np.shape: ", all_labels_np.shape)

    umap_model = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    umap_model.fit(all_features_np)



    # ---------- タスクごとにtransform & 可視化 ----------
    palette = sns.color_palette(n_colors=opt.n_cls)
    marker_list = ['o', 's', '^', 'P', 'X', 'D', 'v', '<', '>', '*', 'H']

    for target_task in range(opt.n_task):

        # target_taskでのクラス数
        n_classes = opt.cls_per_task * (target_task+1)

        task_features = all_features[target_task]
        task_labels = all_labels[target_task]

        task_embed = umap_model.transform(task_features)

        plt.figure(figsize=(5, 5))


        # クラス毎にtask_featuresとtask_lanbelsから取り出して散布図で可視化
        for cls in range(n_classes):
            mask = (cls == task_labels)
            cls_embed = task_embed[mask]

            print("cls_embed.shape: ", cls_embed.shape)
            
            plt.scatter(
                cls_embed[:,0],
                cls_embed[:,1],
                c=palette[cls],
                marker=marker_list[cls//opt.cls_per_task],
                s=10,
                alpha=0.5,
                label=f"Task {target_task}"
            )
        

        ## 
        class_names = [f"Class {i}" for i in range(n_classes)]
        handles = [
            plt.Line2D(
                [], [], 
                marker=marker_list[i//opt.cls_per_task], 
                linestyle='',
                markersize=6, 
                markerfacecolor=palette[i], 
                alpha=0.8,
                label=class_names[i]
            )
            for i in range(n_classes)
        ]
        plt.legend(
            handles=handles,
            title="Classes",
            loc="best",
            fontsize="small",
            title_fontsize="medium",
            ncol=1
        )

        plt.title(f"task {target_task}")
        # plt.xlim(x_min, x_max)
        # plt.ylim(y_min, y_max)
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")

        # plt.xlim(-1, 16)
        # plt.ylim(-1, 16)

        plt.tight_layout()
        plt.savefig(f"./umap_v2/{target_task}.pdf")
        plt.show()






        
        





if __name__ == "__main__":
    main()