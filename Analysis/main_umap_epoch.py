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
    opt.save_path = f'./logs/{opt.method}/{opt.log_name}/annalyze/umap/'


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

    # 特徴とラベルの保存
    all_features = []
    all_labels   = []


    # 各タスクのデータを用いて分析（UMAP）
    for target_task in range(0, opt.n_task):

        # 現在タスクの更新
        opt.target_task = target_task

        # エポック数の決定とUMAPモデルの作成
        if target_task == 0:
            epochs = opt.start_epoch
            # epochs = 1
        else:
            epochs = opt.epoch
        

        # リプレイサンプルいの読み込み
        if opt.target_task == 0:
            replay_indices = np.array([])
        else:
            file_path = f"{opt.log_path}/replay_indices_{opt.target_task}.npy"
            # file_path = f"{opt.log_path}/replay_indices_0.npy"
            replay_indices = np.load(file_path)


        # データローダーの作成（バッファ内のデータも含めて）
        data_loaders  = set_loader(opt, replay_indices)








        # ====================================   最初に現在タスク最後のモデルを読み込んでumapモデルの訓練と可視化を行う  ==================================== #
        # モデルパラメータの読み込み
        ckpt_path = f"{opt.model_path}/model_{target_task:02d}.pth"
        print("ckpt_path: ", ckpt_path)
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

        # 検証用データを使用
        features, labels, layer_outputs = extract_features(opt=opt, model=model, data_loader=data_loaders["val"])

        # Tensor → NumPy（必要なら）
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        n_classes = int(labels.max()) + 1

        # --- エポック0 でだけ UMAP の fit & 軸範囲＆クラス数決定 ---
        umap_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            random_state=42
        ).fit(features.cpu().numpy())
        # umap_model.fit(features.cpu().numpy())
        embed = umap_model.transform(features.cpu().numpy())


        # --- プロット（散布図） ---
        plt.figure(figsize=(5,5))

        # クラス数分の色を用意
        n_classes = int(labels.max()) + 1
        palette = sns.color_palette(n_colors=n_classes)
        # print("palette: ", palette)    # palette:  [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725)]

        batch_size = 500
        n_samples = embed.shape[0]

        # タスク数
        n_tasks = opt.n_cls // opt.cls_per_task
        

        # 利用可能なマーカーリスト（タスクごとに使い分け）
        marker_list = ['o', 's', '^', 'P', 'X', 'D', 'v', '<', '>', '*', 'H']
        marker_cycle = itertools.cycle(marker_list)

        # labelsを参照して task-id を決定
        task_ids = labels // opt.cls_per_task
        # print("labels: ", labels)
        # print("task_ids: ", task_ids)


        for task_id in range(n_tasks):

            task_mask = (task_ids == task_id)
            emb_task = embed[task_mask]
            labels_task = labels[task_mask]
            marker = marker_list[task_id % len(marker_list)]

            plt.scatter(
                emb_task[:,0],
                emb_task[:,1],
                c=[palette[l] for l in labels_task],
                marker=marker,
                s=10,
                alpha=0.5,
                label=f"Task {task_id}"
            )



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

        # plt.xlim(x_min, x_max)
        # plt.ylim(y_min, y_max)
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")

        # plt.xlim(-1, 16)
        # plt.ylim(-1, 16)

        plt.tight_layout()
        plt.savefig(f"./output/{target_task}.pdf")
        plt.show()


        # ====================================   最初に現在タスク最後のモデルを読み込んでumapモデルの訓練と可視化を行う  ==================================== #


        for epoch in range(epochs):

            print("epoch: ", epoch)

            # モデルパラメータの読み込み
            ckpt_path = f"{opt.model_path}/task{opt.target_task:02d}/model_epoch{epoch+1:03d}.pth"
            print("ckpt_path: ", ckpt_path)
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


            # # 訓練用データを使用
            # features, labels, layer_outputs = extract_features(opt=opt, model=model, data_loader=data_loaders["train"])
            # print("1 features.shape: ", features.shape)
            # print("1 labels.shape: ", labels.shape)

            # 検証用データを使用
            features, labels, layer_outputs = extract_features(opt=opt, model=model, data_loader=data_loaders["val"])
            print("1 features.shape: ", features.shape)
            print("1 labels.shape: ", labels.shape)

            # Tensor → NumPy（必要なら）
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()

            n_classes = int(labels.max()) + 1

            # --- エポック0 でだけ UMAP の fit & 軸範囲＆クラス数決定 ---
            # if epoch == 0:
            #     umap_model.fit(features.cpu().numpy())
            #     embed = umap_model.transform(features.cpu().numpy())
            # else:
            #     embed = umap_model.transform(features.cpu().numpy())
            # x_min, x_max = embed[:,0].min(), embed[:,0].max()
            # y_min, y_max = embed[:,1].min(), embed[:,1].max()

            embed = umap_model.transform(features.cpu().numpy())
      


            # --- プロット（散布図） ---
            plt.figure(figsize=(5,5))

            # クラス数分の色を用意
            n_classes = int(labels.max()) + 1
            palette = sns.color_palette(n_colors=n_classes)
            # print("palette: ", palette)    # palette:  [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725)]

            batch_size = 500
            n_samples = embed.shape[0]

            # タスク数
            n_tasks = opt.n_cls // opt.cls_per_task
            

            # 利用可能なマーカーリスト（タスクごとに使い分け）
            marker_list = ['o', 's', '^', 'P', 'X', 'D', 'v', '<', '>', '*', 'H']
            marker_cycle = itertools.cycle(marker_list)

            # labelsを参照して task-id を決定
            task_ids = labels // opt.cls_per_task
            # print("labels: ", labels)
            # print("task_ids: ", task_ids)


            for task_id in range(n_tasks):

                task_mask = (task_ids == task_id)
                emb_task = embed[task_mask]
                labels_task = labels[task_mask]
                marker = marker_list[task_id % len(marker_list)]

                plt.scatter(
                    emb_task[:,0],
                    emb_task[:,1],
                    c=[palette[l] for l in labels_task],
                    marker=marker,
                    s=10,
                    alpha=0.5,
                    label=f"Task {task_id}"
                )



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

            plt.title(f"Epoch {epoch}")
            # plt.xlim(x_min, x_max)
            # plt.ylim(y_min, y_max)
            plt.xlabel("UMAP1")
            plt.ylabel("UMAP2")

            # plt.xlim(-1, 16)
            # plt.ylim(-1, 16)

            plt.tight_layout()
            plt.savefig(f"./output2/task{target_task}/{epoch}.pdf")
            plt.show()


        
        





if __name__ == "__main__":
    main()