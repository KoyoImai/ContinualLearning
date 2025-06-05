import os

import argparse
import numpy as np

import torch

from dataloaders import set_loader
from extract import extract_features
from analysis_tools.analysis_score import Eval_cluster

from util import write_csv_analysis



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

    # rank保存
    all_rank = {}
    alldata_task_rank = {}
    alldata_class_rank = {}

    # 各タスクのデータを用いて分析（SVD）
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


        # # 場合によってはここをコメントアウト
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
        # if opt.use_dp:
        #     features, labels, layer_outputs = extract_features(opt=opt, model=model.module, data_loader=data_loaders["train"])
        # else:
        features, labels, layer_outputs = extract_features(opt=opt, model=model, data_loader=data_loaders["train"])
        print("1 features.shape: ", features.shape)
        print("1 labels.shape: ", labels.shape)
        
        # スコア計算用インスタンス
        agent_score = Eval_cluster(model=model, dataloader=data_loaders["train"], target_module=None, num_features=None)

        # スコアの計算
        sil_score = agent_score._sil_score(features=features.cpu().numpy(), labels=labels.cpu().numpy())
        cal_score = agent_score._cal_score(features=features.cpu().numpy(), labels=labels.cpu().numpy())
        tr_W, tr_B = agent_score._cal_disp(features=features.cpu().numpy(), labels=labels.cpu().numpy())

        print("sil_score: ", sil_score)
        print("cal_score: ", cal_score)
        print("tr_W: ", tr_W)
        print("tr_B: ", tr_B)


        write_csv_analysis(value=sil_score, path=opt.save_path, file_name="sil_score", task_data=opt.target_task, task_model=opt.target_task)
        write_csv_analysis(value=cal_score, path=opt.save_path, file_name="cal_score", task_data=opt.target_task, task_model=opt.target_task)
        write_csv_analysis(value=tr_W, path=opt.save_path, file_name="trW_score", task_data=opt.target_task, task_model=opt.target_task)
        write_csv_analysis(value=tr_B, path=opt.save_path, file_name="trB_score", task_data=opt.target_task, task_model=opt.target_task)



        # # 特徴量を抽出（現在のタスクに含まれるサンプルとリプレイサンプル）
        # features, labels, layer_outputs = extract_features(opt=opt, model=model, data_loader=data_loaders["trainv2"])
        # print("2 features.shape: ", features.shape)
        # print("2 labels.shape: ", labels.shape)

        
        





if __name__ == "__main__":
    main()