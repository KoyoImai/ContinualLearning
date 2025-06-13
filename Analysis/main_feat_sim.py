



import os

import argparse
import numpy as np

import torch

from dataloaders import set_loader
from extract import extract_features
from analysis_tools.analysis_score import Eval_cluster

from util import write_csv_sim, cosine_similarity

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

    # umapのパラメータ
    parser.add_argument('--n_neighbors', type=int, default=100)
    parser.add_argument('--min_dist', type=float, default=0.5)


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
    opt.save_path = f'./logs/{opt.method}/{opt.log_name}/annalyze/feat-sim/'


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

    # 特徴とラベルの保存する辞書
    all_features_dict = {}
    all_labels_dict   = {}


    # 各タスクのデータを用いて分析（UMAP）
    for target_task in range(0, opt.n_task):

        # 現在タスクの更新
        opt.target_task = target_task

        # エポック数の決定
        if target_task == 0:
            # epochs = opt.start_epoch
            epochs = 5
        else:
            # epochs = opt.epoch
            epochs = 5

        

        # リプレイサンプルいの読み込み
        if opt.target_task == 0:
            replay_indices = np.array([])
        else:
            file_path = f"{opt.log_path}/replay_indices_{opt.target_task}.npy"
            # file_path = f"{opt.log_path}/replay_indices_0.npy"
            replay_indices = np.load(file_path)


        # データローダーの作成（バッファ内のデータも含めて）
        data_loaders  = set_loader(opt, replay_indices)


        # 全タスク・全特徴の格納用リストの初期化
        all_features = []
        all_labels = []


        # ---------- 学習済みモデルを呼び出して特徴量を抽出 ----------
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


            # 特徴量とラベルをnumpy配列に変換
            if torch.is_tensor(features):
                features = features.cpu().numpy()
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()


            # 特徴量，ラベルを格納
            all_features.append(features)
            all_labels.append(labels)



        # ---------- 各エポックでの特徴量を格納したリストを辞書に追加 ----------
        all_features_dict[target_task] = all_features
        all_labels_dict[target_task] = all_labels


    # ---------- 各エポックでの特徴量を取り出して特徴量の量の平均を計算 ----------
    avg_feat_dict = {}   # keyはtarget_task，valueは
    
    for target_task in range(0, opt.n_task):
        
        epoch_avg_feat_list = []  # "各エポックにおけるクラス毎の平均特徴量" を格納した辞書をを格納するリスト

        opt.target_task = target_task  # 現在タスクの更新

        # target_task の学習時に獲得した特徴量とラベルを取り出す
        all_features = all_features_dict[target_task]
        all_labels = all_labels_dict[target_task]

        if target_task == 0:
            # epochs = opt.start_epoch
            epochs = 5
        else:
            # epochs = opt.epoch
            epochs = 5


        # "target_task" におけるエポック毎の特徴量とラベルを取り出す
        for epoch in range(epochs):

            print("calcurate average: ", epoch)

            cls_avg_feat_dict = {}

            # epoch目の特徴量とラベルを取り出す
            features = all_features[epoch]
            labels = all_labels[epoch]

            # print("features.shape: ", features.shape)
            # print("labels.shape: ", labels.shape)

            # target_task までに学習したクラス数
            n_classes = opt.cls_per_task * (target_task + 1)

            for cls in range(n_classes):

                # クラスcls の特徴量とラベルのみ取り出す
                cls_mask = (cls == labels)
                cls_feature = features[cls_mask]
                cls_label = labels[cls_mask]

                avg_feature = np.mean(cls_feature, axis=0)
                print("avg_feature.shape: ", avg_feature.shape)

                # クラスcls の平均特徴量を格納
                cls_avg_feat_dict[cls] = avg_feature

            # epoch 目での平均特徴量を格納
            epoch_avg_feat_list.append(cls_avg_feat_dict)
            
        avg_feat_dict[target_task] = epoch_avg_feat_list

    
    # ---------------------  用度整理  ---------------------
    # avg_feat_dict       : キーはtask id，要素はepoch_avg_feat_list
    # epoch_avg_feat_list : インデックスはエポック，要素はcls_avg_feat_dict
    # cls_avg_feat_dict   : キーはクラス番号，要素はクラス番号に対応したデータの平均特徴量



    # ---------- 平均特徴量同士のコサイン類似度を計算（nエポックとn+1エポック） ----------
    for target_task in range(0, opt.n_task):
        
        epoch_avg_feat_list = avg_feat_dict[target_task]

        if target_task == 0:
            # epochs = opt.start_epoch
            epochs = 5
        else:
            # epochs = opt.epoch
            epochs = 5
        

        for epoch in range(epochs-1):

            print("calcurate similarity: ", epoch)

            # 現在タスク（target_task）のクラス数
            num_classes = opt.cls_per_task * target_task

            # nエポック目の平均特徴量とm(n+1)エポック目の平均特徴量を取り出す
            n_cls_avg_feat_dict = epoch_avg_feat_list[epoch]
            m_cls_avg_feat_dict = epoch_avg_feat_list[epoch+1]

            for i in range(num_classes):

                n_avg_feat = n_cls_avg_feat_dict[i]
                m_avg_feat = m_cls_avg_feat_dict[i]
                
                # 形状確認
                # print("n_avg_feat.shape: ", n_avg_feat.shape)
                # print("m_avg_feat.shape: ", m_avg_feat.shape)

                # コサイン類似度の計算
                sim = cosine_similarity(n_avg_feat, m_avg_feat)

                # 形状と値を確認
                # print("sim: ", sim)
                # print("sim.shape: ", sim.shape)

                # 類似度を書き込む
                file_name = f"sim1_model{target_task}_cls{i}"
                write_csv_sim(value=sim, path=opt.save_path, file_name=file_name, epoch=epoch)

    


    # ---------- 平均特徴量同士のコサイン類似度を計算（1エポックとnエポック） ----------
    for target_task in range(0, opt.n_task):
        
        epoch_avg_feat_list = avg_feat_dict[target_task]

        if target_task == 0:
            # epochs = opt.start_epoch
            epochs = 5
        else:
            # epochs = opt.epoch
            epochs = 5
        

        for epoch in range(epochs):

            print("calcurate similarity: ", epoch)

            # 現在タスク（target_task）のクラス数
            num_classes = opt.cls_per_task * target_task

            # nエポック目の平均特徴量とm(n+1)エポック目の平均特徴量を取り出す
            n_cls_avg_feat_dict = epoch_avg_feat_list[0]
            m_cls_avg_feat_dict = epoch_avg_feat_list[epoch]

            for i in range(num_classes):

                n_avg_feat = n_cls_avg_feat_dict[i]
                m_avg_feat = m_cls_avg_feat_dict[i]
                
                # 形状確認
                # print("n_avg_feat.shape: ", n_avg_feat.shape)
                # print("m_avg_feat.shape: ", m_avg_feat.shape)

                # コサイン類似度の計算
                sim = cosine_similarity(n_avg_feat, m_avg_feat)

                # 形状と値を確認
                # print("sim: ", sim)
                # print("sim.shape: ", sim.shape)

                # 類似度を書き込む
                file_name = f"sim2_model{target_task}_cls{i}"
                write_csv_sim(value=sim, path=opt.save_path, file_name=file_name, epoch=epoch)


    # ---------- 平均特徴量同士のコサイン類似度を計算（nタスクの最終エポックとn+1タスクの各エポック） ----------
    for target_task in range(1, opt.n_task):

        before_task = target_task -1
        epoch_avg_feat_list = avg_feat_dict[target_task]
        epoch_avg_feat_list_before = avg_feat_dict[before_task]

        if target_task == 0:
            # epochs = opt.start_epoch
            epochs = 5
        else:
            # epochs = opt.epoch
            epochs = 5


        for epoch in range(epochs):

            print("calcurate similarity: ", epoch)

            # 現在タスク（target_task）のクラス数
            num_classes = opt.cls_per_task * (target_task-1)

            # nエポック目の平均特徴量とm(n+1)エポック目の平均特徴量を取り出す
            n_cls_avg_feat_dict = epoch_avg_feat_list[epoch]
            m_cls_avg_feat_dict = epoch_avg_feat_list_before[-1]


            for i in range(num_classes):

                n_avg_feat = n_cls_avg_feat_dict[i]
                m_avg_feat = m_cls_avg_feat_dict[i]
                
                # 形状確認
                # print("n_avg_feat.shape: ", n_avg_feat.shape)
                # print("m_avg_feat.shape: ", m_avg_feat.shape)

                # コサイン類似度の計算
                sim = cosine_similarity(n_avg_feat, m_avg_feat)

                # 形状と値を確認
                # print("sim: ", sim)
                # print("sim.shape: ", sim.shape)

                # 類似度を書き込む
                file_name = f"sim2_model{target_task}_cls{i}"
                write_csv_sim(value=sim, path=opt.save_path, file_name=file_name, epoch=epoch)





        

        







            





        
        





if __name__ == "__main__":
    main()