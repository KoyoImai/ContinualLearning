import os

import argparse
import numpy as np

import torch


from dataloaders import set_loader
from extract import extract_features
from analysis_tools.analysis_svd_subspace import svd, subspace_similarity, subspace_similarity_diff_task, plot_diff_dist

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
    opt.save_path = f'./logs/{opt.method}/{opt.log_name}/annalyze/sub_space'


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
        model = model.cuda()

    return model




def main():

    # コマンドライン引数の処理
    opt = parse_option()

    # modelの作成
    model = make_setup(opt=opt)

    # replay_indicesの初期化
    replay_indices = None

    # 各タスクのデータを用いて分析（SVD）
    svd_all_class_wise = {}
    svd_all_task_wise = {}
    svd_all = {}
    svd_replay_class_wise = {}
    svd_replay_task_wise = {}

    # 各タスクのデータを用いて分析（SVD）
    for target_task in range(0, opt.n_task):

        # 現在タスクの更新
        opt.target_task = target_task

        # モデルパラメータの読み込み
        # ckpt_path = f"{opt.model_path}/model_{opt.target_task:02d}.pth"
        ckpt_path = f"{opt.model_path}/model_00.pth"
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt['model']
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


        # 特徴量を抽出（これまでのタスクに含まれるすべてのサンプル）
        features, labels = extract_features(opt=opt, model=model, data_loader=data_loaders["train"])
        print("1 features.shape: ", features.shape)  # 1 features.shape:  torch.Size([10000, 512])
        print("1 labels.shape: ", labels.shape)      # 1 labels.shape:  torch.Size([10000])
        

        # SVD（タスク，クラス毎に特徴ベクトルを分割）
        # 各タスク・クラスのsubspaceがどの程度類似しているか
        results = svd(opt=opt, features=features, labels=labels, cls_per_task=opt.cls_per_task, name="alldata")  # タスク毎にSVD
        svd_all_task_wise[target_task] = results
        results = svd(opt=opt, features=features, labels=labels, name="alldata")                                 # クラス毎にSVD
        svd_all_class_wise[target_task] = results

        # SVD（すべての特徴ベクトルを一度にSVD）
        results = svd(opt=opt, features=features, labels=labels, name="alldata", mode="all")  # タスク毎にSVD
        svd_all[target_task] = results

        # 特徴量を抽出（現在のタスクに含まれるサンプルとリプレイサンプル）
        features, labels = extract_features(opt=opt, model=model, data_loader=data_loaders["trainv2"])
        print("2 features.shape: ", features.shape)   # 2 features.shape:  torch.Size([20000, 512])
        print("2 labels.shape: ", labels.shape)       # 2 labels.shape:  torch.Size([20000])
        
        # SVD
        results = svd(opt=opt, features=features, labels=labels, cls_per_task=opt.cls_per_task, name="replay")  # タスク毎にSVD
        svd_replay_task_wise[target_task] = results
        results = svd(opt=opt, features=features, labels=labels, name="replay")                                 # クラス毎にSVD
        svd_replay_class_wise[target_task] = results

    
    # sub spaceの類似度計算
    threshold = 0.9
    dist_acw = {}
    dist_atw = {}
    dist_rcw = {}
    dist_rtw = {}

    # 同じデータに対する特徴表現のsub spaceの類似度
    for t in range(0, opt.n_task-1):

    #     # svd_all_class_wise
    #     # 学習済みデータ全てを対象，クラス毎のsub spaceの類似度計算（svd_all_class_wise）
    #     svd_dict1 = svd_all_class_wise[t]
    #     svd_dict2 = svd_all_class_wise[t+1]

    #     results = subspace_similarity(svd_dict1, svd_dict2, threshold)
    #     dist_acw[t] = results

        # svd_all_task_wise
        # 学習済みデータ全てを対象，タスク毎のsub spaceの類似度計算（svd_all_class_wise）
        svd_dict1 = svd_all_task_wise[t]
        svd_dict2 = svd_all_task_wise[t+1]

        results = subspace_similarity(svd_dict1, svd_dict2, threshold)
        dist_atw[t] = results


    #     # # svd_replay_class_wise
    #     # # 学習済みデータ全てを対象，クラス毎のsub spaceの類似度計算（svd_all_class_wise）
    #     # svd_dict1 = svd_replay_class_wise[t]
    #     # svd_dict2 = svd_replay_class_wise[t+1]

    #     # results = subspace_similarity(svd_dict1, svd_dict2, threshold)
    #     # dist_rcw[t] = results


    #     # # svd_replay_task_wise
    #     # # 学習済みデータ全てを対象，クラス毎のsub spaceの類似度計算（svd_all_class_wise）
    #     # svd_dict1 = svd_replay_task_wise[t]
    #     # svd_dict2 = svd_replay_task_wise[t+1]

    #     # results = subspace_similarity(svd_dict1, svd_dict2, threshold)
    #     # dist_rtw[t] = results
    
    # 可視化
    print("dist_atw: ", dist_atw)
    plot_diff_dist(dist_atw, opt=opt, name="dist_atw")

    
    # task0のデータの特徴表現のsub spaceと他タスクのデータの特徴表現のsub space
    # （アンカーをタスク0＆モデル0の組み合わせの主成分としたとき）
    diff_dist_acw = {}
    diff_dist_atw = {}
    diff_dist_rcw = {}
    diff_dist_rtw = {}

    for t in range(1, opt.n_task):

        # svd_all_task_wise
        # 学習済みデータ全てを対象，タスク毎のsub spaceの類似度計算（svd_all_task_wise）
        svd_dict1 = svd_all_task_wise[0]
        svd1 = svd_dict1[0]
        svd_dict2 = svd_all_task_wise[t]

        results = subspace_similarity_diff_task(svd1, svd_dict2, threshold)
        diff_dist_atw[t] = results
    

    # 可視化
    print("diff_dist_atw: ", diff_dist_atw)
    plot_diff_dist(diff_dist_atw, opt=opt, name="diff_dist_atw")


        










if __name__ == "__main__":
    main()