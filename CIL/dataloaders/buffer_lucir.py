from hmac import new
import random
import math
import select
import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset

from dataloaders.tiny_imagenets import TinyImagenet



def get_features(model, train_loader):

    # 訓練用（ncm_loader）データから全サンプルの特徴とラベルを集めるリスト
    all_features = []
    all_labels = []

    # 平均特徴の計算
    with torch.no_grad():
        for idx, (images, labels) in enumerate(train_loader):

            # gpu上に配置
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            # modelにデータを入力
            y_pred, features = model(x=images, return_feat=True)

            # 特徴量とラベルを保存
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            
    
    # リスト内のテンソルを連結
    all_features = torch.cat(all_features, dim=0)  # shape: [N, feature_dim]
    all_labels = torch.cat(all_labels, dim=0)

    unique_labels = torch.unique(all_labels)
    class_means = {}  # {クラスラベル: 平均特徴}
    
    
    # 保存してある特徴とラベルをもとに各クラスの平均を計算
    for label in unique_labels:
        
        # 該当クラスのサンプルインデックスを抽出
        idxs = (all_labels == label)
        feats = all_features[idxs]
        
        # サンプルごとに特徴を平均
        mean_feat = feats.mean(dim=0, keepdim=True)  # shape: [1, feature_dim]
        class_means[int(label.item())] = mean_feat
    

    # # 辞書のキー（ラベル）が昇順になるようにソートし，平均特徴量を一つのテンソルに変換
    # sorted_labels = sorted(class_means.keys())
    # means_list = [class_means[l] for l in sorted_labels]
    # class_means_tensor = torch.cat(means_list, dim=0)  # shape: [num_classes, feature_dim]
    # print("Computed class means for {} classes.".format(class_means_tensor.shape[0]))

    return all_features, all_labels, class_means



# 平均特徴に近いデータをクラス毎に均等に選択して保存するバッファ
def set_replay_samples_herding(opt, model, prev_indices=None):

    is_training = model.training
    model.eval()

    # データセットの仮作成（ラベルがほしいだけ）
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if opt.dataset == 'cifar10':
        subset_indices = []
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)
    elif opt.dataset == 'cifar100':
        subset_indices = []
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)
    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        val_dataset = TinyImagenet(root=opt.data_folder,
                                    transform=val_transform,
                                    download=True)
        val_targets = val_dataset.targets
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    # 前回タスクのクラスを獲得
    if prev_indices is None:
        prev_indices = []
        observed_classes = list(range(0, opt.target_task*opt.cls_per_task))
    else:

        # 前回タスクのクラス範囲
        observed_classes = list(range(max(opt.target_task-1, 0)*opt.cls_per_task, (opt.target_task)*opt.cls_per_task))

    # 確認済みのクラス（前回タスク）がない場合終了
    if len(observed_classes) == 0:
        return prev_indices
    
    # これまで学習した全てのクラス
    all_learned_classes = list(range(0, opt.target_task*opt.cls_per_task))

    # 確認済みクラスのインデックスを獲得
    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()
    

    # バッファ内データのクラスとバッファ外データのクラスのデータセット
    combined_indices = observed_indices + prev_indices
    select_dataset =  Subset(val_dataset, combined_indices)
    # print("len(train_dataset): ", len(train_dataset))

    # バッファ内データのクラスとバッファ外データのクラスのデータローダー
    select_loader = torch.utils.data.DataLoader(
        select_dataset, batch_size=500, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)
    # print("len(select_loader): ", len(select_loader))

    # 各データの特徴量，ラベル，平均特徴量を獲得
    all_features, all_labels, class_means = get_features(model, select_loader)

    # 各データのインデックス
    original_indices = np.array(select_dataset.indices)

    # 各クラスのデータ選択数
    num_all_learned_classes = len(all_learned_classes)
    samples_per_class = int(opt.mem_size / num_all_learned_classes)
    remainder = opt.mem_size % num_all_learned_classes        # 余りのサンプル数

    # 最終的に返すインデックス集合
    final_selected_indices = []

    for i, c in enumerate(all_learned_classes):
        
        # クラス順序 i が remainder より小さい場合は1サンプル多く割り当てる
        allocated_samples = samples_per_class + (1 if i < remainder else 0)

        class_mask = (all_labels == c)

        # クラスcに属するサンプルの「select_loader内での順番」上の位置を取得
        class_positions = torch.nonzero(class_mask).flatten().cpu().numpy()

        # クラスcのデータのデータセット上のインデックスを獲得
        class_orig_indices = original_indices[class_positions]

        # クラス平均とのL2距離を計算
        feats = all_features[class_mask]  # 該当クラスの特徴量 [n_samples, feature_dim]
        mean_feat = class_means[c]          # [1, feature_dim]

        # L2距離（ノルム）の計算（各サンプルと平均との差）
        distances = torch.norm(feats - mean_feat, p=2, dim=1)  # shape: [n_samples]

        # 距離が小さい順（平均に近い順）にソートするためのインデックス
        sorted_order = torch.argsort(distances)

        # ソート済みの元のインデックス
        sorted_class_orig_indices = class_orig_indices[sorted_order.cpu().numpy()]

        # 各クラスから選ぶべき数が，実際のサンプル数より多い場合は，存在するサンプルを全て選ぶ
        num_to_select = min(len(sorted_class_orig_indices), allocated_samples)
        selected_for_class = sorted_class_orig_indices[:num_to_select].tolist()
        
        print("Class {}: selected {}/{} samples.".format(c, len(selected_for_class), allocated_samples))
        final_selected_indices.extend(selected_for_class)

    

    model.is_training = is_training
    # print("len(final_selected_indices): ", len(final_selected_indices))
    # assert None

    return final_selected_indices



