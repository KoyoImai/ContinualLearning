import os

import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch


from dataloaders import set_loader
from extract import extract_features
from analysis_tools.analysis_svd import svd
from analysis_tools.analysis_pca import compute_pca, compute_mean_drift

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

    # その他
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--num_workers", type=int, default=8)

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
    opt.save_path = f'./logs/{opt.method}/{opt.log_name}/annalyze/pca-drift'


    # ディレクトリ作成
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)



    return opt




def make_setup(opt):

    if opt.method in ["co2l", "supcon-joint"]:

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

    else:

        assert False
    

    if torch.cuda.is_available():
        model = model.cuda()

    return model




def main():

    # コマンドライン引数の処理
    opt = parse_option()

    # modelの作成
    model = make_setup(opt=opt)

    # replay_indicesの初期化
    replay_indices = None

    # 各タスクのデータを用いて分析（PCA）
    U_all_class_wise = []
    U_all_task_wise = []
    U_replay_class_wise = []
    U_replay_task_wise = []
    for target_task in range(0, opt.n_task):

        # 現在タスクの更新
        opt.target_task = target_task

        # モデルパラメータの読み込み
        ckpt_path = f"{opt.model_path}/model_{opt.target_task:02d}.pth"
        # ckpt_path = f"{opt.model_path}/model_00.pth"
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['model']
        model.load_state_dict(state_dict)

        # リプレイサンプルの読み込み
        if opt.target_task == 0:
            replay_indices = np.array([])
        else:
            file_path = f"{opt.log_path}/replay_indices_{opt.target_task}.npy"
            # file_path = f"{opt.log_path}/replay_indices_0.npy"
            replay_indices = np.load(file_path)

        # データローダーの作成（バッファ内のデータも含めて）
        data_loaders  = set_loader(opt, replay_indices)

        # 特徴量を抽出（これまでのタスクに含まれるすべてのサンプル）
        features, labels = extract_features(opt=opt, model=model, data_loader=data_loaders["train"])

        # PCA による主成分の取り出し-全データ（クラス毎に主成分を取り出し，task idをkeyとした辞書で返却）
        pca_dict, pca_list = compute_pca(opt=opt, features=features, labels=labels, n_components=20, cls_per_task=1)
        U_all_class_wise += [pca_dict]

        pca_dict, pca_list = compute_pca(opt=opt, features=features, labels=labels, n_components=20, cls_per_task=opt.cls_per_task)
        U_all_task_wise += [pca_dict]

        # 特徴量を抽出（現在のタスクに含まれるサンプルとリプレイサンプル）
        features, labels = extract_features(opt=opt, model=model, data_loader=data_loaders["trainv2"])

        # PCA による主成分の取り出し-現タスクのデータ+リプレイ（クラス毎に主成分を取り出し，task idをkeyとした辞書で返却）
        pca_dict, pca_list = compute_pca(opt=opt, features=features, labels=labels, n_components=20, cls_per_task=1)
        U_replay_class_wise += [pca_dict]

        pca_dict, pca_list = compute_pca(opt=opt, features=features, labels=labels, n_components=20, cls_per_task=opt.cls_per_task)
        U_replay_task_wise += [pca_dict]
    
    
    # U_allとU_replayをもとに主成分の変化を確認 
    drift_all_class = []
    drift_all_task  = []
    drift_rep_class = []
    drift_rep_task  = []

    for t in range(0, opt.n_task-1):

        # task t と task t+1 のモデルの主成分のdriftを計算
        drift_ac = compute_mean_drift(U_all_class_wise[t], U_all_class_wise[t+1], opt=opt)
        drift_at = compute_mean_drift(U_all_task_wise[t],  U_all_task_wise[t+1], opt=opt)
        drift_rc = compute_mean_drift(U_replay_class_wise[t], U_replay_class_wise[t+1], opt=opt)
        drift_rt = compute_mean_drift(U_replay_task_wise[t],  U_replay_task_wise[t+1], opt=opt)

        drift_all_class.append(drift_ac)
        drift_all_task.append(drift_at)
        drift_rep_class.append(drift_rc)
        drift_rep_task.append(drift_rt)

    # ========================
    # drift をプロット
    # ========================
    task_ids = list(range(1, opt.n_task))

    plt.figure(figsize=(10, 5))
    plt.plot(task_ids, drift_all_class, marker='o', label='All Data (Class-wise)', color='C0')
    plt.plot(task_ids, drift_rep_class, marker='s', label='Replay + Current (Class-wise)', color='C1')
    plt.plot(task_ids, drift_all_task,  marker='^', label='All Data (Task-wise)', color='C2')
    plt.plot(task_ids, drift_rep_task,  marker='D', label='Replay + Current (Task-wise)', color='C3')

    plt.xlabel("Task t")
    plt.ylabel("PCA Drift (1 - mean cosine similarity)")
    plt.title("Drift in Principal Components across Tasks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig("./output_supcon-joint.pdf")
    plt.savefig(f"./output_{opt.method}.pdf")

    plt.show()








if __name__ == "__main__":
    main()