import os
import math
import random
import numpy as np


import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset

from dataloaders.tiny_imagenets import TinyImagenet




def set_replay_samples_cclis_wo_is(opt, prev_indices=None, prev_importance_weight=None, prev_score=None):

    # is_training = model.training
    # model.eval()

    class IdxDataset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.dataset)
        def __getitem__(self, idx):
            return self.indices[idx], self.dataset[idx]

    # construct data loader
    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if opt.dataset == 'cifar10':
        subset_indices = []
        print(opt.data_folder)
        print(os.getcwd())
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)

    elif opt.dataset == 'tiny-imagenet':
        subset_indices = []
        val_dataset = TinyImagenet(root=opt.data_folder,
                                    transform=val_transform,
                                    download=True)
        val_targets = np.array(val_dataset.targets)

    elif opt.dataset == 'cifar100':
        subset_indices = []
        print(opt.data_folder)
        print(os.getcwd())
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         transform=val_transform,
                                         download=True)
        val_targets = np.array(val_dataset.targets)

    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    prev_indices_len = 0

    if prev_indices is None:
        prev_indices, prev_importance_weight = [], []
        observed_classes = list(range(0, opt.target_task*opt.cls_per_task))
    else:
        shrink_size = ((opt.target_task - 1) * opt.mem_size / opt.target_task)
        if len(prev_indices) > 0:
            unique_cls = np.unique(val_targets[prev_indices])
            _prev_indices = prev_indices
            prev_indices_len = len(prev_indices)
            prev_indices = []
            prev_weight = prev_importance_weight 
            prev_importance_weight = []

            for c in unique_cls:
                mask = val_targets[_prev_indices] == c

                size_for_c = shrink_size / len(unique_cls)
                p = size_for_c - (shrink_size // len(unique_cls))  

                if random.random() < p:
                    size_for_c = math.ceil(size_for_c)
                else:
                    size_for_c = math.floor(size_for_c)

                # scoreを重みとしてサンプルのインデックスを取り出す
                # store_index = torch.multinomial(torch.tensor(prev_score[:prev_indices_len])[mask],
                #                                 min(len(torch.tensor(prev_score[:prev_indices_len])[mask]), size_for_c),
                #                                 replacement=False)  # score tensor [old_samples_num] 

                # ランダムにサンプルのインデックスを取り出す
                indices_pool = torch.where(torch.tensor(val_targets[_prev_indices]) == c)[0]
                rand_perm = torch.randperm(len(indices_pool))[:min(len(indices_pool), size_for_c)]
                store_index = indices_pool[rand_perm]


                prev_indices += torch.tensor(_prev_indices)[mask][store_index].tolist()

                prev_cur_weight = torch.tensor(prev_score[:prev_indices_len])[mask]

                prev_importance_weight += (prev_cur_weight / prev_cur_weight.sum())[store_index].tolist()

            print(np.unique(val_targets[prev_indices], return_counts=True))
        observed_classes = list(range(max(opt.target_task-1, 0)*opt.cls_per_task, (opt.target_task)*opt.cls_per_task))

    if len(observed_classes) == 0:
        return prev_indices, prev_importance_weight, val_targets
    
    # 観測直後のタスクのクラス（1タスク前のクラス）
    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()

    # ラベルの獲得
    val_observed_targets = val_targets[observed_indices]
    val_unique_cls = np.unique(val_observed_targets)


    selected_observed_indices = []
    selected_observed_importance_weight = []
    for c_idx, c in enumerate(val_unique_cls):
        size_for_c_float = ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) / (len(val_unique_cls) - c_idx))
        p = size_for_c_float -  ((opt.mem_size - len(prev_indices) - len(selected_observed_indices)) // (len(val_unique_cls) - c_idx))
        if random.random() < p:
            size_for_c = math.ceil(size_for_c_float)
        else:
            size_for_c = math.floor(size_for_c_float)

        # 特定クラスcのみを取り出すためのマスク
        mask = val_targets[observed_indices] == c
        
        # prev_scoreをもとにサンプル毎に重みづけをして保存するサンプルを選択
        # store_index = torch.multinomial(torch.tensor(prev_score[prev_indices_len:])[mask], size_for_c, replacement=False)

        # ランダムにサンプルのインデックスを取り出す
        indices_pool = torch.where(torch.tensor(val_targets[observed_indices]) == c)[0]
        rand_perm = torch.randperm(len(indices_pool))[:size_for_c]
        store_index = indices_pool[rand_perm]

        # 選択されたサンプルのインデックスを蓄積
        # selected_observed_indices += torch.tensor(observed_indices)[mask][store_index].tolist()

        indices_pool = torch.where(torch.tensor(val_targets[observed_indices]) == c)[0]
        rand_perm = torch.randperm(len(indices_pool))[:size_for_c]
        store_index = indices_pool[rand_perm]
        selected_indices = torch.tensor(observed_indices)[store_index]  # ← グローバル変換
        selected_observed_indices += selected_indices.tolist()

        # 特定クラスcに属する全サンプルのスコアを取り出す（提案分布）
        observed_cur_weight = torch.tensor(prev_score[prev_indices_len:])[mask] 
        
        # スコアを正規化
        observed_normalized_weight = observed_cur_weight / observed_cur_weight.sum() 

        # 保存するサンプルのスコアのみを取り出す
        selected_observed_importance_weight += observed_normalized_weight[store_index].tolist()  

    print(np.unique(val_targets[selected_observed_indices], return_counts=True))
    print(selected_observed_importance_weight)

    return prev_indices + selected_observed_indices, prev_importance_weight + selected_observed_importance_weight, val_targets