import os
import copy
import random
import argparse
import numpy as np
import logging


from scipy import optimize
import torch
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset
import torch.optim.lr_scheduler as lr_scheduler


from util import seed_everything, save_model
from dataloaders.make_buffer import set_buffer
from dataloaders.make_dataloader import set_loader
from trains.main_train import train
from preprocesses.main_preprocess import pre_process
from postprocesses.main_postprocess import post_process



def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # 手法
    parser.add_argument('--method', type=str, default="",
                        choices=['er', 'co2l', 'gpm', 'lucir', 'fs-dgpm', 'cclis', 'supcon', 'supcon-joint', 'simclr',
                                 'cclis-bw','cclis-wo', 'cclis-wo-replay', 'cclis-wo-ss', 'cclis-wo-is', 'cclis-rfr'])

    # logの名前（実行毎に変えてね）
    parser.add_argument('--log_name', type=str, default="practice")


    # データセット周り
    parser.add_argument('--data_folder', type=str, default='/home/kouyou/Datasets/', help='path to custom dataset')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet', 'path'], help='dataset')


    # 最適化手法
    parser.add_argument('--learning_rate', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    
    # 学習条件
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--vanilla_batch_size', type=int, default=500)


    # classifierの学習条件(Co2Lなど線形分類で後から評価する手法用)
    parser.add_argument('--linear_epochs', type=int, default=100)
    parser.add_argument('--linear_lr', type=float, default=0.1)
    parser.add_argument('--linear_momentum', type=float, default=0.9)
    parser.add_argument('--linear_weight_decay', type=float, default=0)
    parser.add_argument('--linear_batch_size', type=int, default=256)

    # 継続学習的設定
    parser.add_argument('--mem_size', type=int, default=500)
    parser.add_argument('--mem_type', type=str, default="ring",
                        choices=["reservoir", "ring", "herding"])
    
    # 使用するモデル
    parser.add_argument("--model", type=str, default="resnet18")

    # 手法毎のハイパラ（共通）
    parser.add_argument("--temp", type=float, default=2)
    parser.add_argument("--lamda", type=float, default=5)

    # 手法毎のハイパラ（co2l）
    parser.add_argument('--current_temp', type=float, default=0.2)
    parser.add_argument('--past_temp', type=float, default=0.1)
    parser.add_argument('--distill_power', type=float, default=0.1)

    # 手法毎のハイパラ（gpm & fs-dgpm）
    parser.add_argument('--threshold', type=float, default=0.965)

    # 手法毎のハイパラ（lucir）
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--dist", type=float, default=0.5)
    parser.add_argument("--lw_mr", type=float, default=1)

    # 手法毎のハイパラ（cclis）
    parser.add_argument('--distill_type', type=str, default="PRD")
    parser.add_argument('--max_iter', type=int, default=5,
                        help='iterations of the score computing')
    parser.add_argument('--learning_rate_prototypes', type=float, default=0.01)
    parser.add_argument('--cosine', default=False, action='store_true')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    # parser.add_argument('--warm', action='store_true',
    #                     help='warm-up for large batch training')

    # 手法毎のハイパラ（fs-dgpm）
    # parser.add_argument('--inner_batches', type=int, default=2)
    parser.add_argument('--inner_steps', type=int, default=2)
    parser.add_argument('--freeze_bn', default=False, action='store_true')
    parser.add_argument('--second_order', default=False, action='store_true', help='')
    parser.add_argument('--mem_batch_size', type=int, default=64)
    parser.add_argument('--grad_clip_norm', type=float, default=2.0, help='Clip the gradients by this value')
    parser.add_argument('--sharpness', default=False, action='store_true', help='')
    parser.add_argument('--eta1', type=float, default=1e-2, help='update step size of weight perturbation')
    parser.add_argument('--eta2', type=float, default=1e-2, help='learning rate of lambda(soft weight for basis)')
    parser.add_argument('--fsdgpm_method', type=str, default='xdgpm', choices=['xdgpm', 'dgpm', 'xgpm'])
    parser.add_argument('--thres_add', type=float, default=0.003, help='thres_add')
    parser.add_argument('--thres_last', type=float, default=0.99999999999, help='thres_last')
    parser.add_argument('--lam_init', type=float, default=1.0, help='temperature for sigmoid')
    parser.add_argument('--tmp', type=float, default=10, help='temperature for sigmoid')

    parser.add_argument('--bw_lambd', default=0.0050, type=float, metavar='L',
                        help='weight on off-diagonal terms')

    # 手法ごとのハイパラ（cclis + α）
    parser.add_argument('--rfr_power', type=float, default=0.1)


    # その他の条件
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--date', type=str, default="2001_05_02")
    parser.add_argument('--earlystop', default=False, action='store_true', help='')
    parser.add_argument('--epoch_save', default=False, action='store_true')   # エポック毎にモデルを保存

    parser.add_argument('--offline', default=False, action='store_true')

    opt = parser.parse_args()

    return opt


def setup_logging(opt):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),                   # コンソール出力
            logging.FileHandler(f"{opt.explog_path}/experiment.log", mode="w")  # ファイルに記録（上書きモード）
        ]
    )


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

    # モデルの保存，実験記録などの保存先パス
    if opt.data_folder is None:
        opt.data_folder = '~/data/'
    opt.model_path = f'./logs/{opt.method}/{opt.log_name}/model/'      # modelの保存先
    opt.explog_path = f'./logs/{opt.method}/{opt.log_name}/exp_log/'   # 実験記録の保存先
    opt.mem_path = f'./logs/{opt.method}/{opt.log_name}/mem_log/'      # リプレイバッファ内の保存先

    # ディレクトリ作成
    if not os.path.isdir(opt.model_path):
        os.makedirs(opt.model_path)
    if not os.path.isdir(opt.explog_path):
        os.makedirs(opt.explog_path)
    if not os.path.isdir(opt.mem_path):
        os.makedirs(opt.mem_path)
    



def make_setup(opt):

    from dataloaders.make_dataloader import set_loader

    method_tools = {}

    print("opt.method: ", opt.method)

    # 手法毎にモデル構造，損失関数，最適化手法を作成
    if opt.method == "er":

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_er import BackboneResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = BackboneResNet(name='resnet18', head='linear', feat_dim=opt.n_cls, seed=opt.seed)
        print("model: ", model)
        # assert False
        model2 = None
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
        method_tools = {"optimizer": optimizer}
    
    elif opt.method == "co2l":

        from losses.loss_co2l import SupConLoss
        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_co2l import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed)
        model2 = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed)
        criterion = SupConLoss(temperature=opt.temp)
        optimizer = optim.SGD(model.parameters(),
                                lr=opt.learning_rate,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
        method_tools = {"optimizer": optimizer}

    elif opt.method == "gpm":
        
        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_gpm import ResNet18
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = ResNet18(nf=64, nclass=opt.n_cls, seed=opt.seed)

        print("model: ", model)
        # assert False
        model2 = None
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
        
        method_tools = {"feature_list": [], "threshold": None, "feature_mat": [], "optimizer": optimizer}
    
    elif opt.method == "lucir":

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_lucir import BackboneResNet
        elif opt.dataset in ["imagemet"]:
            assert False

        model = BackboneResNet(name='resnet18', head='linear', feat_dim=opt.cls_per_task, seed=opt.seed)
        print("model: ", model)

        model2 = BackboneResNet(name='resnet18', head='linear', feat_dim=opt.cls_per_task, seed=opt.seed)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
        
        method_tools = {"cur_lamda": opt.lamda, "optimizer": optimizer}

    elif opt.method == "fs-dgpm":
        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            if opt.model == "resnet18":
                from models.resnet_cifar_fsdgpm import ResNet18
                model = ResNet18(nf=64, nclass=opt.n_cls, seed=opt.seed, opt=opt)
                print("model: ", model)
                model2 = None
            else:
                assert False
        elif opt.dataset in ["imagemet"]:
            assert False
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
        method_tools = {"optimizer": optimizer}
    
    elif opt.method == "scr":
        # from losses.loss_co2l import SupConLoss
        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_scr import SCRResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SCRResNet(name='resnet18', head='mlp', feat_dim=128)
        model2 = SupConResNet(name='resnet18', head='mlp', feat_dim=128)
        criterion = SupConLoss(temperature=0.07)
        optimizer = optim.SGD(model.parameters(),
                                lr=opt.learning_rate,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
        assert False

    elif opt.method in ["cclis", "cclis-wo-replay", "cclis-wo-ss", "cclis-wo-is"]:

        from losses.loss_cclis import ISSupConLoss

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_cclis import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        model2 = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        criterion = ISSupConLoss(temperature=opt.temp, opt=opt)

        # optimizer = optim.SGD(model.parameters(),
        #                         lr=opt.learning_rate,
        #                         momentum=opt.momentum,
        #                         weight_decay=opt.weight_decay)


        if 'prototypes.weight' in model.state_dict().keys():
            optimizer = optim.SGD([
                            {'params': model.encoder.parameters()},
                            {'params': model.head.parameters()},
                            {'params': model.prototypes.parameters(), 'lr': opt.learning_rate_prototypes},
                            ],
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
        else:
            learning_rate =  opt.learning_rate
            optimizer = optim.SGD(model.parameters(),
                            lr=learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
        method_tools = {"optimizer": optimizer, "importance_weight": None, "score": None,
                        "score_mask": None, "subset_sample_num": None, "post_loader": None, "val_targets": None}

    elif opt.method in ["cclis-wo"]:

        from losses.loss_cclis_wo import ISSupConLoss

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_cclis import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        model2 = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        criterion = ISSupConLoss(temperature=opt.temp, opt=opt)

        if 'prototypes.weight' in model.state_dict().keys():
            optimizer = optim.SGD([
                            {'params': model.encoder.parameters()},
                            {'params': model.head.parameters()},
                            {'params': model.prototypes.parameters(), 'lr': opt.learning_rate_prototypes},
                            ],
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
        else:
            learning_rate =  opt.learning_rate
            optimizer = optim.SGD(model.parameters(),
                            lr=learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
        method_tools = {"optimizer": optimizer, "importance_weight": None, "score": None,
                        "score_mask": None, "subset_sample_num": None, "post_loader": None, "val_targets": None}

    elif opt.method in ["supcon", "supcon-joint"]:

        from losses.loss_supcon import SupConLoss
        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_co2l import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed)
        model2 = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed)
        criterion = SupConLoss(temperature=opt.temp)
        optimizer = optim.SGD(model.parameters(),
                                lr=opt.learning_rate,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
        method_tools = {"optimizer": optimizer}
    
    elif opt.method == "simclr":

        from losses.loss_simclr import ContrastiveLoss
        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_co2l import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed)
        model2 = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed)
        criterion = ContrastiveLoss(temperature=opt.temp)
        optimizer = optim.SGD(model.parameters(),
                                lr=opt.learning_rate,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
        method_tools = {"optimizer": optimizer}

    # CCLIS + Barlow Twins
    elif opt.method in ["cclis-bw"]:

        from losses.loss_cclisbw import ISSupConLoss, BarlowTwinsLoss

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_cclis import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False

        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        model2 = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        criterion = ISSupConLoss(temperature=opt.temp, opt=opt)
        criterion_bw = BarlowTwinsLoss(opt=opt, size=128)
        if torch.cuda.is_available():
            criterion_bw = criterion_bw.cuda()

        if 'prototypes.weight' in model.state_dict().keys():
            optimizer = optim.SGD([
                            {'params': model.encoder.parameters()},
                            {'params': model.head.parameters()},
                            {'params': model.prototypes.parameters(), 'lr': opt.learning_rate_prototypes},
                            ],
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
        else:
            learning_rate =  opt.learning_rate
            optimizer = optim.SGD(model.parameters(),
                            lr=learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
        
        method_tools = {"optimizer": optimizer, "importance_weight": None, "score": None, "criterion_bw": criterion_bw,
                        "score_mask": None, "subset_sample_num": None, "post_loader": None, "val_targets": None}

    # CCLIS + RFR
    elif opt.method in ["cclis-rfr"]:

        from losses.loss_cclis import ISSupConLoss

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_cclis import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        model2 = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        criterion = ISSupConLoss(temperature=opt.temp, opt=opt)
        criterion_rfr = None

        # optimizer = optim.SGD(model.parameters(),
        #                         lr=opt.learning_rate,
        #                         momentum=opt.momentum,
        #                         weight_decay=opt.weight_decay)


        if 'prototypes.weight' in model.state_dict().keys():
            optimizer = optim.SGD([
                            {'params': model.encoder.parameters()},
                            {'params': model.head.parameters()},
                            {'params': model.prototypes.parameters(), 'lr': opt.learning_rate_prototypes},
                            ],
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
        else:
            learning_rate =  opt.learning_rate
            optimizer = optim.SGD(model.parameters(),
                            lr=learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
        method_tools = {"optimizer": optimizer, "importance_weight": None, "score": None,
                        "score_mask": None, "subset_sample_num": None, "post_loader": None, "val_targets": None}
        
    else:
        assert False

    # gpu上に配置
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        if model2 is not None:
            model2 = model2.cuda()
    
    return model, model2, criterion, method_tools


def make_scheduler(opt, epochs, dataloader, method_tools):

    optimizer = method_tools["optimizer"]

    if opt.method in ["gpm"]:
        scheduler = None
    
    elif opt.method in ["co2l", "simclr"]:
        print("len(dataloader): ", len(dataloader))
        if opt.target_task == 0:
            total_steps = opt.start_epoch * len(dataloader)
            pct_start = (10 * len(dataloader)) / total_steps
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=opt.learning_rate, total_steps=total_steps, pct_start=pct_start, anneal_strategy='cos')
        else:
            total_steps = opt.epochs * len(dataloader)
            pct_start = (10 * len(dataloader)) / total_steps
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=opt.learning_rate, total_steps=total_steps, pct_start=pct_start, anneal_strategy='cos')
    
    # elif opt.method in ["cclis", "supcon", "supcon-joint", "cclis-wo", "cclis-wo-ss", "cclis-wo-is"]:
    elif opt.method in ["supcon", "supcon-joint", "cclis-wo-ss", "cclis-wo-is", "er"]:
        print("len(dataloader): ", len(dataloader))
        if opt.target_task == 0:
            total_steps = opt.start_epoch * len(dataloader)
            pct_start = (10 * len(dataloader)) / total_steps
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=opt.learning_rate, total_steps=total_steps, pct_start=pct_start, anneal_strategy='cos')
        else:
            total_steps = opt.epochs * len(dataloader)
            pct_start = (10 * len(dataloader)) / total_steps
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=opt.learning_rate, total_steps=total_steps, pct_start=pct_start, anneal_strategy='cos')
    
    elif opt.method == "lucir":
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    
    elif opt.method in ["fs-dgpm"]:
        total_steps = epochs * len(dataloader)
        if opt.target_task == 0:
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=opt.learning_rate, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=opt.learning_rate, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos')
    
    elif opt.method in ["cclis", "cclis-bw", "cclis-wo", "cclis-rfr"]:   # 別の方法でschedulerを実装
        scheduler = None
    else:
        assert False

    return scheduler, method_tools


def main():

    # コマンドライン引数の処理
    opt = parse_option()

    # print("opt.learning_rate: ", opt.learning_rate)
    # assert False

    # 乱数のシード固定（既存のコードに追加）
    seed_everything(opt.seed)

    # logの名前
    opt.log_name = f"{opt.log_name}_{opt.method}_{opt.mem_type}{opt.mem_size}_{opt.dataset}_seed{opt.seed}_date{opt.date}"

    # データローダ作成の前処理
    preparation(opt)

    # loggerの設定
    setup_logging(opt=opt)
    logging.info("Experiment started")

    # modelの作成，損失関数の作成，Optimizerの作成
    model, model2, criterion, method_tools = make_setup(opt)
    print("model: ", model)
    # param = model.encoder.conv1.weight.data
    # print("param: ", param)
    

    # バッファ内データのインデックス
    replay_indices = None

    # タスク毎の学習エポック数
    original_epochs = opt.epochs

    # 各タスクの学習
    for target_task in range(0, opt.n_task):

        # 現在タスクの更新
        opt.target_task = target_task
        print('Start Training current task {}'.format(opt.target_task))
        logging.info('Start Training current task {}'.format(opt.target_task))

        # 教師モデル（model2）のパラメータを生徒モデルのパラメータでコピー
        model2 = copy.deepcopy(model)

        # リプレイバッファ内にあるデータのインデックスを獲得
        replay_indices, method_tools = set_buffer(opt, model, prev_indices=replay_indices, method_tools=method_tools)
        # print("main.py replay_indices: ", replay_indices)

        # バッファ内データのインデックスを保存（検証や分析時に読み込むため）
        np.save(
          os.path.join(opt.mem_path, 'replay_indices_{target_task}.npy'.format(target_task=target_task)),
          np.array(replay_indices))
        
        # データローダーの作成（バッファ内のデータも含めて）
        dataloader, subset_indices, method_tools = set_loader(opt, replay_indices, method_tools)

        # 検証や分析用にデータを保存
        np.save(
          os.path.join(opt.mem_path, 'subset_indices_{target_task}.npy'.format(target_task=target_task)),
          np.array(subset_indices))


        # 訓練前にエポック数を設定（初期エポックだけエポック数を変える場合に必要）
        if target_task == 0 and opt.start_epoch is not None:
            opt.epochs = opt.start_epoch
        else:
            opt.epochs = original_epochs

        # # schedulerの作成
        # scheduler = make_scheduler(opt=opt, epochs=opt.epochs, optimizer=optimizer, dataloader=dataloader["train"])

        # タスク開始後の前処理（gpmなどの前処理が必要な手法のため）
        method_tools, model, model2 = pre_process(opt=opt, model=model, model2=model2, dataloader=dataloader, method_tools=method_tools)

        # schedulerの作成
        scheduler, method_tools = make_scheduler(opt=opt, epochs=opt.epochs, dataloader=dataloader["train"], method_tools=method_tools)

        # 訓練を実行
        for epoch in range(1, opt.epochs+1):

            # 学習 & 検証
            train(opt=opt, model=model, model2=model2, criterion=criterion,
                  optimizer=method_tools["optimizer"], scheduler=scheduler, dataloader=dataloader,
                  epoch=epoch, method_tools=method_tools)
            
            # epoch毎にパラメータを保存
            if opt.epoch_save:
                dir_path = f"{opt.model_path}/task{opt.target_task:02d}"
                file_path = f"{dir_path}/model_epoch{epoch:03d}.pth"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                save_model(model, method_tools["optimizer"], opt, opt.epochs, file_path)

            
        # タスク終了後の後処理（gpmなどの後処理が必要な手法のため）
        method_tools, model2 = post_process(opt=opt, model=model, model2=model2, dataloader=dataloader, criterion=criterion, method_tools=method_tools, replay_indices=replay_indices)

        # 保存（opt.model_path）
        file_path = f"{opt.model_path}/model_{opt.target_task:02d}.pth"
        save_model(model, method_tools["optimizer"], opt, opt.epochs, file_path)

        # print("method_tools['score_mask']: ", method_tools["score_mask"])
        # assert False

    


if __name__ == "__main__":
    main()