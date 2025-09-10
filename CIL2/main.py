
import os
import copy
import argparse
import logging
import numpy as np


import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


from dataloaders import set_buffer, set_loader
from preprocesses import pre_process
from postprocesses import post_process
from train import train


from util import seed_everything, save_model





def parse_option():
    
    parser = argparse.ArgumentParser('argument for training')

    # 基本的な実験設定
    parser.add_argument("--log_name", type=str, default="test")
    parser.add_argument('--method', type=str, default='cclis', choices=['er', 'co2l', 'cclis', 'prco'])

    # データセット関連
    parser.add_argument('--data_folder', type=str, default='/home/kouyou/Datasets/', help='path to custom dataset')
    parser.add_argument('--data_order', type=str, default="original")
    parser.add_argument('--dataset', type=str, default="cifar100", choices=["cifar10", "cifar100", "tiny-imagenet", "imagenet"])


    # 最適化設定
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.03)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)


    parser.add_argument('--linear_batch_size', type=int, default=128)
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
    parser.add_argument('--temp_co2l', type=float, default=0.5)
    parser.add_argument('--current_temp', type=float, default=0.2)
    parser.add_argument('--past_temp', type=float, default=0.1)
    parser.add_argument('--distill_power', type=float, default=0.1)
    parser.add_argument('--not_asym', default=False, action='store_true')

    # cclisのハイパーパラメータ
    parser.add_argument('--temp_cclis', type=float, default=0.5)
    parser.add_argument('--wo_is', action='store_true')
    parser.add_argument('--learning_rate_prototypes', type=float, default=0.01)
    parser.add_argument('--cosine', default=False, action='store_true')
    parser.add_argument('--distill_type', type=str, default="PRD")
    parser.add_argument('--max_iter', type=int, default=5, help='iterations of the score computing')

    # prcoのハイパーパラメータ（一部のハイパラはCo2L・CCLISと共通）
    parser.add_argument('--temp_prco', type=float, default=0.5)
    

    # その他の設定
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--date', type=str, default='2024_0101')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=50)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--epoch_save', default=False, action='store_true')   # エポック毎にモデルを保存


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



def setup_logging(opt):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),                   # コンソール出力
            logging.FileHandler(f"{opt.explog_path}/experiment.log", mode="w")  # ファイルに記録（上書きモード）
        ]
    )


def make_setup(opt):

    if opt.method in ["er"]:

        assert False

    elif opt.method in ["co2l"]:

        from losses.loss_co2l import SupConLoss, ContrastiveLoss

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_co2l import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed)
        model2 = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed)
        criterion = SupConLoss(temperature=opt.temp_co2l, not_asym=opt.not_asym)

        optimizer = optim.SGD(model.parameters(),
                                lr=opt.learning_rate,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
        method_tools = {"optimizer": optimizer}


    elif opt.method in ["cclis"]:

        from losses.loss_cclis import ISSupConLoss
        from losses.loss_co2l import ContrastiveLoss

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_cclis import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        

        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        model2 = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        criterion = ISSupConLoss(temperature=opt.temp_cclis, opt=opt)


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
        method_tools = {"optimizer": optimizer}

    elif opt.method in ["prco"]:

        from losses.loss_prco import ProtoSupConLoss

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_prco import SupConResNet
        elif opt.dataset in ["imagenet"]:
            assert False

        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        model2 = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        criterion = ProtoSupConLoss(temperature=opt.temp_prco, opt=opt)

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
        method_tools = {"optimizer": optimizer}
    
    else:

        assert False
    

    if torch.cuda.is_available():
        model = model.cuda()
        model2 = model2.cuda()
        criterion = criterion.cuda()

        model = torch.nn.DataParallel(model)
        model2 = torch.nn.DataParallel(model2)

    return model, model2, criterion, method_tools
    



def make_scheduler(opt, epochs, dataloader, method_tools):

    optimizer = method_tools["optimizer"]

    if opt.method in ['er']:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.1)

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
    
    elif opt.method in ["cclis"]:
        scheduler = None

    elif opt.method in ["prco"]:
        scheduler = None
    
    else:
        assert False

    return scheduler, method_tools




def main():

    # コマンドライン引数の処理
    opt = parse_option()

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
    # print("model: ", model)
    # assert False


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

        # バッファ内データのインデックスを保存（検証や分析時に読み込むため）
        np.save(
          os.path.join(opt.mem_path, 'replay_indices_{target_task}.npy'.format(target_task=target_task)),
          np.array(replay_indices))
    
        # データローダーの作成（バッファ内のデータも含めて）
        dataloader, subset_indices = set_loader(opt, model, replay_indices, method_tools)

        # 検証や分析用にデータを保存
        np.save(
          os.path.join(opt.mem_path, 'subset_indices_{target_task}.npy'.format(target_task=target_task)),
          np.array(subset_indices))


        # 訓練前にエポック数を設定（初期エポックだけエポック数を変える場合に必要）
        if target_task == 0 and opt.start_epoch is not None:
            opt.epochs = opt.start_epoch
        else:
            opt.epochs = original_epochs


        # タスク開始後の前処理（gpmなどの前処理が必要な手法のため）
        method_tools, model, model2 = pre_process(opt=opt, model=model, model2=model2, dataloader=dataloader, method_tools=method_tools)

        # schedulerの作成
        scheduler, method_tools = make_scheduler(opt=opt, epochs=opt.epochs, dataloader=dataloader["train"], method_tools=method_tools)

        
        # ランダム初期化のモデルを保存
        if opt.target_task == 0:
            file_path = f"{opt.model_path}/model_random.pth"
            # save_model(model, method_tools["optimizer"], opt, opt.epochs, file_path)
            save_model(model, method_tools["optimizer"], opt, opt.epochs, file_path)

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
        post_process(opt=opt, model=model, model2=model2, dataloader=dataloader, criterion=criterion, method_tools=method_tools, replay_indices=replay_indices)

        # 保存（opt.model_path）
        file_path = f"{opt.model_path}/model_{opt.target_task:02d}.pth"
        # save_model(model, method_tools["optimizer"], opt, opt.epochs, file_path)
        save_model(model, method_tools["optimizer"], opt, opt.epochs, file_path)


if __name__ == "__main__":
    main()

