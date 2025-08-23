
import os
import copy
import h5py
import time
import random

import argparse
import numpy as np

import torch


from util import seed_everything

from models.resnet_cifar_co2l import LinearClassifier

import tools_losslandscape.projection as proj
import tools_losslandscape.scheduler as scheduler
import tools_losslandscape.evaluation as evaluation
from tools_losslandscape.make_dataloaders import make_dataloader
from tools_losslandscape.landscape import get_weights, name_direction_file, setup_direction, load_directions
from tools_losslandscape.landscape import set_states, set_weights
import tools_losslandscape.plot_2D as plot_2D


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # 手法の決定
    parser.add_argument("--method", type=str, default="cclis")

    # データセットの決定
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument('--data_folder', type=str, default='/home/kouyou/Datasets/', help='path to custom dataset')
    parser.add_argument("--cls_list", required=True, nargs="*", type=int)

    # モデルの種類，事前学習済みモデルのパス
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument('--model_file', type=str, default='', help='path to the trained model file')
    parser.add_argument('--model_file2', type=str, default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--model_file3', type=str, default='', help='use (model_file3 - model_file) as the ydirection')
    parser.add_argument('--classifier_file', type=str, default='', help='path to the trained model file')
    parser.add_argument('--classifier_file2', type=str, default='', help='use (model_file2 - model_file) as the xdirection')
    parser.add_argument('--classifier_file3', type=str, default='', help='use (model_file3 - model_file) as the ydirection')
    
    # 手法毎のハイパラ（co2l，cclis）
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--use_classifier", action='store_true')

    # 手法毎のハイパラ（cclis）
    parser.add_argument('--wo_is', default=False, action='store_true')

    # 手法毎のハイパラ
    parser.add_argument('--not_asym', default=False, action='store_true')

    # ハイパラ
    parser.add_argument('--batch_size', type=int, default=512)


    # ---- runtime / system ----
    parser.add_argument('--cuda', action='store_true', help='use CUDA if available')
    parser.add_argument('--threads', type=int, default=2, help='num dataloader workers')
    parser.add_argument('--ngpu', type=int, default=1, help='num GPUs per process')

    # ---- 可視化する範囲・解像度の指定 ----
    parser.add_argument('--x', type=str, default='-1:1:51', help='xmin:xmax:xnum')
    parser.add_argument('--y', type=str, default=None, help='ymin:ymax:ynum (2D when set)')

    # ---- 方向ベクトルの正規化（フィルタ正規化） ----
    parser.add_argument('--xnorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--ynorm', default='', help='direction normalization: filter | layer | weight')
    parser.add_argument('--xignore', default='', help='ignore bias and BN parameters: biasbn')
    parser.add_argument('--yignore', default='', help='ignore bias and BN parameters: biasbn')


    # ---- direction parameters ----
    parser.add_argument('--dir_file', default='', help='specify the name of direction file, or the path to an eisting direction file')
    parser.add_argument('--dir_type', type=str, default='random', choices=['random', 'weights', 'states'],
                        help='random directions or states-difference for x-axis')
    # parser.add_argument('--base_ckpt', type=str, default=None, help='path to base checkpoint (theta*)')
    # parser.add_argument('--second_ckpt', type=str, default=None, help='path to second checkpoint (for states direction)')
    parser.add_argument('--skip_bn_bias', action='store_true', help='ignore BN/bias in direction')
    parser.add_argument('--norm', type=str, default='filter', choices=['filter','layer','weight','none'],
                        help='direction normalization (filter recommended)')
    parser.add_argument('--same_dir', action='store_true', default=False, help='use the same random direction for both x-axis and y-axis')
    parser.add_argument('--idx', default=0, type=int, help='the index for the repeatness experiment')
    parser.add_argument('--surf_file', default='', help='customize the name of surface file, could be an existing file.')


    # ---- evaluation control ----
    parser.add_argument('--max_batches', type=int, default=None, help='avg over at most K batches per grid point')
    parser.add_argument('--save_png', type=str, default='landscape_2d.png', help='output figure path')


    # plot parameters
    parser.add_argument('--proj_file', default='', help='the .h5 file contains projected optimization trajectory.')
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')

    # その他
    parser.add_argument("--seed", type=int, default=777)


    opt = parser.parse_args()


    # parse x/y ranges into numbers
    try:
        opt.xmin, opt.xmax, opt.xnum = [float(a) for a in opt.x.split(':')]
        if opt.y:
            opt.ymin, opt.ymax, opt.ynum = [float(a) for a in opt.y.split(':')]
        else:
            opt.ymin = opt.ymax = opt.ynum = None
    except Exception as e:
        raise ValueError(f'Bad format for --x/--y. Use like "-1:1:51". ({e})')


    return opt


def preparation(opt):

    # データセット毎にタスク数・タスク毎のクラス数を決定
    if opt.dataset == 'cifar10':
        opt.n_cls = 10
        opt.cls_per_task = 2
        opt.size = 32
    if opt.dataset == 'cifar100':
        opt.n_cls = 100
        opt.cls_per_task = 5
        opt.size = 32
    elif opt.dataset == 'tiny-imagenet':
        opt.n_cls = 200
        opt.cls_per_task = 20
        opt.size = 64
    else:
        pass


# モデルと損失関数の設計
def make_setup(opt):

    if opt.method == "co2l":

        from losses.loss_co2l import SupConLoss

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_co2l import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed)
        criterion = SupConLoss(temperature=opt.temp, not_asym=opt.not_asym)
        
    
    elif opt.method == "cclis":

        from losses.loss_cclis import ISSupConLoss

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_cclis import SupConResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = SupConResNet(name='resnet18', head='mlp', feat_dim=128, seed=opt.seed, opt=opt)
        criterion = ISSupConLoss(temperature=opt.temp, opt=opt)
        
    
    elif opt.method == "er":

        if opt.dataset in ["cifar10", "cifar100", "tiny-imagenet"]:
            from models.resnet_cifar_er import BackboneResNet
        elif opt.dataset in ["imagemet"]:
            assert False
        
        model = BackboneResNet(name='resnet18', head='linear', feat_dim=opt.n_cls, seed=opt.seed)
        criterion = torch.nn.CrossEntropyLoss()
    

    return model, criterion


# ---------------------------------------------
# 可視化結果の出力ファイル名を決定
# ---------------------------------------------
def name_surface_file(args, dir_file, use_classifier=False):
    # skip if surf_file is specified in args
    if args.surf_file:
        return args.surf_file

    # use args.dir_file as the perfix
    surf_file = dir_file

    # resolution
    surf_file += '_[%s,%s,%d]' % (str(args.xmin), str(args.xmax), int(args.xnum))
    if args.y:
        surf_file += 'x[%s,%s,%d]' % (str(args.ymin), str(args.ymax), int(args.ynum))

    # dataloder parameters
    if args.raw_data: # without data normalization
        surf_file += '_rawdata'
    if args.data_split > 1:
        surf_file += '_datasplit=' + str(args.data_split) + '_splitidx=' + str(args.split_idx)

    if not use_classifier:
        return surf_file + ".h5"
    elif use_classifier:
        return surf_file + "_classifier.h5"


# ---------------------------------------------
# 可視化結果の出力ファイルを作成
# ---------------------------------------------
def setup_surface_file(args, surf_file, dir_file):
    # skip if the direction file already exists
    if os.path.exists(surf_file):
        f = h5py.File(surf_file, 'r')
        if (args.y and 'ycoordinates' in f.keys()) or 'xcoordinates' in f.keys():
            f.close()
            print ("%s is already set up" % surf_file)
            return

    f = h5py.File(surf_file, 'a')
    f['dir_file'] = dir_file

    # Create the coordinates(resolutions) at which the function is evaluated
    xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
    f['xcoordinates'] = xcoordinates

    if args.y:
        ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.ynum))
        f['ycoordinates'] = ycoordinates
    f.close()

    return surf_file




def crunch(surf_file, net, w, s, d, dataloader, loss_key, acc_key, criterion, args):

    # ファイルを開く
    f = h5py.File(surf_file, 'r+')
    losses, accuracies = [], []

    # 座標と結果配列を用意
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None
    
    # 結果がなければ -1 で埋める
    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        
        f[loss_key] = losses
        f[acc_key] = accuracies
    
    # 結果が存在すれば読み込み
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]
    
    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates)

    print('Computing %d values'% (len(inds)))
    start_time = time.time()
    total_sync = 0.0


    # 損失を計算
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':
            set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
        elif args.dir_type == 'states':
            set_states(net.module if args.ngpu > 1 else net, s, d, coord)
        
        # Record the time to compute the loss value
        loss_start = time.time()

        if args.method == "cclis":
            loss, acc = evaluation.evaluation_cclis(net, criterion, dataloader, args)
        elif args.method == "co2l":
            loss, acc = evaluation.evaluation_co2l(net, criterion, dataloader, args)


        # 対処損失の場合精度が出ないので代わりに0埋め
        if acc is None:
            acc = 0.0

        loss_compute_time = time.time() - loss_start


        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc


        # Send updated plot data to the master node
        syc_start = time.time()
        syc_time = time.time() - syc_start
        total_sync += syc_time

        f[loss_key][:] = losses
        f[acc_key][:] = accuracies
        f.flush()

        print('Evaluating  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))


    total_time = time.time() - start_time
    print('Done!! Total time: %.2f Sync: %.2f' % (total_time, total_sync))

    f.close()






def crunch_with_classifier(surf_file, net, classifier, w, w_classifier, s, s_classifier, d, d_classifier, dataloader, loss_key, acc_key, args):

    # 交差エントロピー損失
    criterion = torch.nn.CrossEntropyLoss()


    # ファイルを開く
    f = h5py.File(surf_file, 'r+')
    losses, accuracies = [], []

    # 座標と結果配列を用意
    xcoordinates = f['xcoordinates'][:]
    ycoordinates = f['ycoordinates'][:] if 'ycoordinates' in f.keys() else None
    
    # 結果がなければ -1 で埋める
    if loss_key not in f.keys():
        shape = xcoordinates.shape if ycoordinates is None else (len(xcoordinates),len(ycoordinates))
        losses = -np.ones(shape=shape)
        accuracies = -np.ones(shape=shape)
        
        f[loss_key] = losses
        f[acc_key] = accuracies
    
    # 結果が存在すれば読み込み
    else:
        losses = f[loss_key][:]
        accuracies = f[acc_key][:]
    
    # Generate a list of indices of 'losses' that need to be filled in.
    # The coordinates of each unfilled index (with respect to the direction vectors
    # stored in 'd') are stored in 'coords'.
    inds, coords, inds_nums = scheduler.get_job_indices(losses, xcoordinates, ycoordinates)

    print('Computing %d values'% (len(inds)))
    start_time = time.time()
    total_sync = 0.0


    # 損失を計算
    for count, ind in enumerate(inds):
        # Get the coordinates of the loss value being calculated
        coord = coords[count]

        # Load the weights corresponding to those coordinates into the net
        if args.dir_type == 'weights':

            # modelとclassifierの重みを変更
            set_weights(net.module if args.ngpu > 1 else net, w, d, coord)
            set_weights(classifier.module if args.ngpu > 1 else classifier, w_classifier, d_classifier, coord)

        elif args.dir_type == 'states':
            set_states(net.module if args.ngpu > 1 else net, s, d, coord)
            set_states(classifier.module if args.ngpu > 1 else classifier, s_classifier, d_classifier, coord)

        
        # Record the time to compute the loss value
        loss_start = time.time()

        if args.method == "cclis":
            loss, acc = evaluation.evaluation_classifier_cclis(net, classifier, criterion, dataloader, args)
        elif args.method == "co2l":
            loss, acc = evaluation.evaluation_classifier_co2l(net, classifier, criterion, dataloader, args)


        # 対処損失の場合精度が出ないので代わりに0埋め
        if acc is None:
            acc = 0.0

        loss_compute_time = time.time() - loss_start


        # Record the result in the local array
        losses.ravel()[ind] = loss
        accuracies.ravel()[ind] = acc


        # Send updated plot data to the master node
        syc_start = time.time()
        syc_time = time.time() - syc_start
        total_sync += syc_time

        f[loss_key][:] = losses
        f[acc_key][:] = accuracies
        f.flush()

        print('Evaluating  %d/%d  (%.1f%%)  coord=%s \t%s= %.3f \t%s=%.2f \ttime=%.2f \tsync=%.2f' % (
                count, len(inds), 100.0 * count/len(inds), str(coord), loss_key, loss,
                acc_key, acc, loss_compute_time, syc_time))


    total_time = time.time() - start_time
    print('Done!! Total time: %.2f Sync: %.2f' % (total_time, total_sync))

    f.close()






def main():

    # コマンドライン引数の処理
    opt = parse_option()

    print("opt.x: ", opt.x)
    print("opt.y: ", opt.y)

    # -----------------------------
    # データローダ作成の前処理
    # -----------------------------
    preparation(opt)


    # -----------------------------
    # seed値の固定
    # -----------------------------
    seed_everything(opt.seed)


    # -----------------------------
    # model classifier 損失関数の用意
    # モデルはresnet18，損失関数は交差エントロピー・対照損失（・蒸留損失など）が主な対象
    # -----------------------------
    model, criterion = make_setup(opt=opt)
    model2, _ = make_setup(opt=opt)           # --model_file2 が存在する場合，その重みパラメータを読み込むようのモデル
    model3, _ = make_setup(opt=opt)           # --model_file3 が存在する場合，その重みパラメータを読み込むようのモデル

    if opt.method in ["co2l", "cclis"] and opt.use_classifier:
        classifier = LinearClassifier(name="resnet18", num_classes=10, seed=opt.seed)
        classifier2 = LinearClassifier(name="resnet18", num_classes=10, seed=opt.seed)
        classifier3 = LinearClassifier(name="resnet18", num_classes=10, seed=opt.seed)


    # -----------------------------
    # 事前学習済みモデルの読み込み
    # 読み込んだmodelのパラメータを保存しておき，摂動を加えた後に元に戻せるように保管
    # -----------------------------
    ckpt = torch.load(opt.model_file, map_location='cpu')
    state_dict = ckpt['model']
    if isinstance(model, torch.nn.DataParallel):
        new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    w = get_weights(model)                    # 初期パラメータ（Tensorのリスト）
    s = copy.deepcopy(model.state_dict())     # state_dict も保持（DeepCopy）

    if opt.method in ["co2l", "cclis"] and opt.use_classifier:

        ckpt = torch.load(opt.classifier_file, map_location='cpu')
        state_dict = ckpt['model']
        classifier.load_state_dict(state_dict)

        w_classifier = get_weights(classifier)                    # 初期パラメータ（Tensorのリスト）
        s_classifier = copy.deepcopy(classifier.state_dict())     # state_dict も保持（DeepCopy）

    # -----------------------------
    # 方向ベクトルの準備，ファイル保存
    # -----------------------------
    dir_file = name_direction_file(opt)                       # 方向ファイル名の決定
    dir_file_classifier = name_direction_file(opt, use_classifier=opt.use_classifier)

    setup_direction(opt, dir_file, model, model2, model3)     # 方向ファイルの生成
    if opt.use_classifier:
        setup_direction(opt, dir_file_classifier, classifier, classifier2, classifier3)     # 方向ファイルの生成 for classifier


    surf_file = name_surface_file(opt, dir_file)
    setup_surface_file(opt, surf_file, dir_file)

    # if opt.use_classifier:
    #     surf_file_classifier = name_surface_file(opt, dir_file, True)



    # -----------------------------
    # 方向ベクトルの読み込み
    # -----------------------------
    d = load_directions(dir_file)
    d_classifier = load_directions(dir_file_classifier)

    # calculate the consine similarity of the two directions
    if len(d) == 2:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d[0]), proj.nplist_to_tensor(d[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)

    if len(d_classifier) == 2:
        similarity = proj.cal_angle(proj.nplist_to_tensor(d_classifier[0]), proj.nplist_to_tensor(d_classifier[1]))
        print('cosine similarity between x-axis and y-axis: %f' % similarity)
    

    # -----------------------------
    # データローダーの作成
    # -----------------------------
    dataloaders = make_dataloader(opt)


    # モデルと損失関数をgpu上に配置
    if torch.cuda.is_available():
        model = model.cuda().eval()
        criterion = criterion.cuda()

        if opt.use_classifier:
            classifier = classifier.cuda().eval()

    # データローダーを取り出し
    trainloader, val_loader = dataloaders


    #--------------------------------------------------------------------------
    # Start the computation
    #--------------------------------------------------------------------------
    if not opt.use_classifier:
        crunch(surf_file, model, w, s, d, trainloader, 'train_loss', 'train_acc', criterion, opt)
    elif opt.use_classifier:
        crunch_with_classifier(surf_file, model, classifier, w, w_classifier, s, s_classifier, d, d_classifier, trainloader, 'train_loss', 'train_acc', opt)

    
    #--------------------------------------------------------------------------
    # 可視化を実行
    #--------------------------------------------------------------------------
    if opt.plot:
        if opt.y and opt.proj_file:
            plot_2D.plot_contour_trajectory(surf_file, dir_file, opt.proj_file, 'train_loss', opt.show)
        elif opt.y:
            plot_2D.plot_2d_contour(surf_file, 'train_loss', opt.vmin, opt.vmax, opt.vlevel, opt.show)
        else: 
            plot_1D.plot_1d_loss_err(surf_file, opt.xmin, opt.xmax, opt.loss_max, opt.log, opt.show)




if __name__ == "__main__":
    main()



