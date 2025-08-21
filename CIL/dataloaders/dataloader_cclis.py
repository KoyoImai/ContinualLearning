
import copy
from operator import methodcaller
import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import Sampler, RandomSampler

from dataloaders.tiny_imagenets import TinyImagenet


class IS_Subset(Dataset):
    def __init__(self, dataset, indices, IS_weight):
        self.dataset = dataset
        self.indices = indices
        self.weight = IS_weight

    def __getitem__(self, idx):
        index = self.indices[idx]
        weight = self.weight[idx]
        image, label = self.dataset[index]

        # print("[DEBUG] __getitem__ called")  # ← 絶対呼ばれるはず
        return image, label, weight, index

    def __len__(self):
        return len(self.indices)
    

# class IS_Subset(Subset):
#     """
#     Defines dataset with importance sampling weight.
#     """
#     def __init__(self, dataset, indices, IS_weight) -> None:
#         super().__init__(dataset, indices)
#         self.weight = IS_weight
        
#     def __getitem__(self, idx):
#         if isinstance(idx, list):
#             index = [self.indices[i] for i in idx]
#             weight = [self.weight[i] for i in idx]
#         else:
#             index = self.indices[idx]
#             weight = self.weight[idx]

#         return super().__getitem__(idx) + (weight, index) 
    
#     def __len__(self):
#         return super().__len__()



# 各タスク，各クラスのサンプル数を指定してミニバッチを作成したい
class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset

        self.batch_size = batch_size  # list 
        self.number_of_datasets = len(dataset.datasets) 

        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])
        self.dataset_len = sum([len(cur_dataset) for cur_dataset in self.dataset.datasets])

    def __len__(self):
        return self.dataset_len

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset) 
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1] 
        step = sum(self.batch_size) 

        samples_to_grab, epoch_samples = self.batch_size, self.dataset_len  
        # print('epoch_samples', epoch_samples)

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab[i]):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration: 
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        break

                final_samples_list.extend(cur_samples)

        return iter(final_samples_list)



# 訓練用cifar10
def set_loader_cclis_cifar10(opt, normalize, replay_indices, method_tools, training=True):

    importance_weight = method_tools['importance_weight']
    # print("importance_weight: ", importance_weight)
    # assert False

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print('target_classes', target_classes)

    # 
    subset_indices = []
    subset_importance_weight = []

    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
        tc_num = (np.array(_train_dataset.targets) == tc).sum()
        
        subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list

    _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

    if len(replay_indices) > 0 and training:
        prev_dataset = IS_Subset(_train_dataset, replay_indices, importance_weight)
        cur_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        train_dataset = ConcatDataset([prev_dataset, cur_dataset])


    else:
        _subset_indices += replay_indices
        _subset_importance_weight += importance_weight

        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

    subset_indices += replay_indices
    subset_importance_weight += importance_weight

    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
    replay_sample_num = uc[np.argsort(uk)]


    if len(replay_indices) > 0 and training: 
        train_batch_size_list = [int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list))), 
                                 opt.batch_size - int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list)))]
        
        print('train_batch_size', train_batch_size_list)
        train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=train_batch_size_list)
        print('len_data', [len(cur_dataset) for cur_dataset in train_sampler.dataset.datasets])
    else:
        train_sampler = None
        
    if training:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)


    else:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True)
        print('no separate sampler')
    

    # Samplerが正しく機能しているかの確認
    # for b, (imgs, labels, _, idxs) in enumerate(train_loader):

    #     # 2) クラス別枚数
    #     uniq, cnt = labels.unique(return_counts=True)
    #     print("  class_counts:", dict(zip(uniq.tolist(), cnt.tolist())))

    
    return train_loader, subset_indices, replay_sample_num


# ある１クラスのサンプルのみを含む検証用データローダーの作成cifar10
def set_grad_loader_cclis_cifar10(opt, train, normalize, method_tools):

    importance_weight = method_tools['importance_weight']

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    train_loaders = []

    for tc in range(opt.n_cls):

        subset_indices = []
        subset_importance_weight = []

        _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                          train=train,
                                          transform=train_transform,
                                          download=True)

        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
        tc_num = (np.array(_train_dataset.targets) == tc).sum()
        
        subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list

        _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)
        bsz = int(len(train_dataset) / 5)
        print("bsz: ", bsz)

        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=bsz, shuffle=True,
                            num_workers=opt.num_workers, pin_memory=True)
        
        train_loaders += [train_loader]

    return train_loaders



# ある１タスクのサンプルのみを含むデータローダーの作成cifar10
def set_gradtask_loader_cclis_cifar10(opt, train, normalize, method_tools):

    importance_weight = method_tools['importance_weight']

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    train_loaders = []

    for task_id in range(opt.n_task):

        target_classes = list(range(task_id*opt.cls_per_task, (task_id+1)*opt.cls_per_task))

        subset_indices = []
        subset_importance_weight = []

        _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                            transform=train_transform,
                                            download=True)
        
        for tc in target_classes:
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
            tc_num = (np.array(_train_dataset.targets) == tc).sum()
            subset_importance_weight += list(np.ones(tc_num) / tc_num)
            
        


        _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)
        bsz = int(len(train_dataset) / 5)
        print("bsz: ", bsz)

        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=bsz, shuffle=True,
                            num_workers=opt.num_workers, pin_memory=True)
        
        train_loaders += [train_loader]

    return train_loaders



# 全てのサンプルを含んだデータローダーを返す
def set_grad_loader_cclis_cifar10_v2(opt, normalize, replay_indices, method_tools, training=True):

    importance_weight = method_tools['importance_weight']

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print('target_classes', target_classes)

    # 
    subset_indices = []
    subset_importance_weight = []

    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                      train=training,
                                      transform=train_transform,
                                      download=True)
    
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
        tc_num = (np.array(_train_dataset.targets) == tc).sum()
        
        subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list

    _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

    if len(replay_indices) > 0 and training:
        prev_dataset = IS_Subset(_train_dataset, replay_indices, importance_weight)
        cur_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        train_dataset = ConcatDataset([prev_dataset, cur_dataset])

    else:
        _subset_indices += replay_indices
        _subset_importance_weight += importance_weight

        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

    subset_indices += replay_indices
    subset_importance_weight += importance_weight

    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
    replay_sample_num = uc[np.argsort(uk)]


    if len(replay_indices) > 0 and training: 
        train_batch_size_list = [int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list))), 
                                 opt.batch_size - int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list)))]
        
        print('train_batch_size', train_batch_size_list)
        train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=train_batch_size_list)
        print('len_data', [len(cur_dataset) for cur_dataset in train_sampler.dataset.datasets])
    else:
        train_sampler = None
        
    if training:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)


    else:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True)
        print('no separate sampler')
    
    return None


# 現在タスクのデータ+リプレイバッファのデータを含むデータローダーを返す（基本的に訓練用データローダーを同じ）
def set_gradreplay_loader_cclis_cifar10(opt, normalize, replay_indices, method_tools, training=True):

    importance_weight = method_tools['importance_weight']

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print('target_classes', target_classes)

    # 
    subset_indices = []
    subset_importance_weight = []

    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                      train=training,
                                      transform=train_transform,
                                      download=True)
    
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
        tc_num = (np.array(_train_dataset.targets) == tc).sum()
        
        subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list

    _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

    if len(replay_indices) > 0 and training:
        prev_dataset = IS_Subset(_train_dataset, replay_indices, importance_weight)
        cur_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        train_dataset = ConcatDataset([prev_dataset, cur_dataset])

    else:
        _subset_indices += replay_indices
        _subset_importance_weight += importance_weight

        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

    subset_indices += replay_indices
    subset_importance_weight += importance_weight

    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
    replay_sample_num = uc[np.argsort(uk)]


    if len(replay_indices) > 0 and training: 
        train_batch_size_list = [int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list))), 
                                 opt.batch_size - int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list)))]
        
        print('train_batch_size', train_batch_size_list)
        train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=train_batch_size_list)
        print('len_data', [len(cur_dataset) for cur_dataset in train_sampler.dataset.datasets])
    else:
        train_sampler = None
        
    if training:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=500, shuffle=(train_sampler is None),
                            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=False)


    else:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=500, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
        print('no separate sampler')
    
    return train_loader














# cifar100データセットのクラスをsuperclassに倣って変更
# fine(0..99) -> coarse(0..19)
groups = [
    # 0: aquatic mammals
    [4, 30, 55, 72, 95],
    # 1: fish
    [1, 32, 67, 73, 91],
    # 2: flowers
    [54, 62, 70, 82, 92],
    # 3: food containers
    [9, 10, 16, 28, 61],
    # 4: fruit and vegetables
    [0, 51, 53, 57, 83],
    # 5: household electrical devices
    [22, 39, 40, 86, 87],
    # 6: household furniture
    [5, 20, 25, 84, 94],
    # 7: insects
    [6, 7, 14, 18, 24],
    # 8: large carnivores
    [3, 42, 43, 88, 97],
    # 9: large man-made outdoor things
    [12, 17, 37, 68, 76],
    # 10: large natural outdoor scenes
    [23, 33, 49, 60, 71],
    # 11: large omnivores and herbivores
    [15, 19, 21, 31, 38],
    # 12: medium-sized mammals
    [34, 63, 64, 66, 75],
    # 13: non-insect invertebrates
    [26, 45, 77, 79, 99],
    # 14: people
    [2, 11, 35, 46, 98],
    # 15: reptiles
    [27, 29, 44, 78, 93],
    # 16: small mammals
    [36, 50, 65, 74, 80],
    # 17: trees
    [47, 52, 56, 59, 96],
    # 18: vehicles 1
    [8, 13, 48, 58, 90],
    # 19: vehicles 2
    [41, 69, 81, 85, 89],
]








# 訓練用cifar100
def set_loader_cclis_cifar100(opt, normalize, replay_indices, method_tools, training=True):

    importance_weight = method_tools['importance_weight']

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクで学習対象となるクラスリスト
    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print('target_classes', target_classes)

    subset_indices = []
    subset_importance_weight = []

    # cifar100データセット
    _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)
    
    # cifar100データセットのクラスを入れ替え
    if opt.data_order == "sparse2coarse":

        # cifar100のsuperclassに基づいて，クラスラベルを書き換える
        # 水性哺乳類に含まれるクラス4, 30, 55, 72, 95を0, 1, 2, 3, 4に書き換える
        # 魚に含まれるクラス1, 32, 67, 73, 91を5, 6, 7, 8, 9に書き換える
        # 残りも同様に，，，，
        remap = {}
        for g_idx, fine_list in enumerate(groups):
            base = g_idx * 5
            for offset, old_label in enumerate(fine_list):
                remap[old_label] = base + offset
        
        # _train_dataset.targets を新ラベルに書き換え
        _train_dataset.targets = [remap[int(t)] for t in _train_dataset.targets]

    elif opt.data_order == "original":
        print("data order random")

    
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
        tc_num = (np.array(_train_dataset.targets) == tc).sum()

        subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list
    
    _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

    if len(replay_indices) > 0 and training:
        prev_dataset = IS_Subset(_train_dataset, replay_indices, importance_weight)
        cur_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        train_dataset = ConcatDataset([prev_dataset, cur_dataset])
    else:
        _subset_indices += replay_indices
        _subset_importance_weight += importance_weight
        print('_subset_indices length', len(_subset_indices))
        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

    subset_indices += replay_indices
    subset_importance_weight += importance_weight

    print('dataset length', len(_train_dataset), len(train_dataset))        
    print('Dataset size: {}'.format(len(subset_indices)))

    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
    replay_sample_num = uc[np.argsort(uk)]

    if len(replay_indices) > 0 and training: 
        train_batch_size_list = [int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list))), 
                                 opt.batch_size - int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list)))]
        
        print('train_batch_size', train_batch_size_list)
        train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=train_batch_size_list)
        print('len_data', [len(cur_dataset) for cur_dataset in train_sampler.dataset.datasets])
    else:
        train_sampler = None
        
    if training:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)


    else:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True)
        print('no separate sampler')
    
    return train_loader, subset_indices, replay_sample_num


# ある１クラスのサンプルのみを含む検証用データローダーの作成cifar100
def set_grad_loader_cclis_cifar100(opt, train, normalize, method_tools):

    importance_weight = method_tools['importance_weight']

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    

    train_loaders = []

    for tc in range(opt.n_cls):

        subset_indices = []
        subset_importance_weight = []

        _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                           train=train,
                                           transform=train_transform,
                                           download=True)

        # cifar100データセットのクラスを入れ替え
        if opt.data_order == "sparse2coarse":

            remap = {}
            for g_idx, fine_list in enumerate(groups):
                base = g_idx * 5
                for offset, old_label in enumerate(fine_list):
                    remap[old_label] = base + offset
            
            # _train_dataset.targets を新ラベルに書き換え
            _train_dataset.targets = [remap[int(t)] for t in _train_dataset.targets]

        elif opt.data_order == "original":
            print("data order random")
        
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]

        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
        tc_num = (np.array(_train_dataset.targets) == tc).sum()
        
        subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list

        _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)
        bsz = int(len(train_dataset) / 5)
        print("bsz: ", bsz)

        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=bsz, shuffle=True,
                            num_workers=opt.num_workers, pin_memory=True)
        
        train_loaders += [train_loader]

    return train_loaders



# ある１タスクのサンプルのみを含む検証用データローダーの作成cifar100
def set_gradtask_loader_cclis_cifar100(opt, train, normalize, method_tools):

    importance_weight = method_tools['importance_weight']

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    train_loaders = []

    for task_id in range(opt.n_task):

        target_classes = list(range(task_id*opt.cls_per_task, (task_id+1)*opt.cls_per_task))

        subset_indices = []
        subset_importance_weight = []

        _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                           train=train,
                                           transform=train_transform,
                                           download=True)

        # cifar100データセットのクラスを入れ替え
        if opt.data_order == "sparse2coarse":

            # cifar100のsuperclassに基づいて，クラスラベルを書き換える
            # 水性哺乳類に含まれるクラス4, 30, 55, 72, 95を0, 1, 2, 3, 4に書き換える
            # 魚に含まれるクラス1, 32, 67, 73, 91を5, 6, 7, 8, 9に書き換える
            # 残りも同様に，，，，
            remap = {}
            for g_idx, fine_list in enumerate(groups):
                base = g_idx * 5
                for offset, old_label in enumerate(fine_list):
                    remap[old_label] = base + offset
            
            # _train_dataset.targets を新ラベルに書き換え
            _train_dataset.targets = [remap[int(t)] for t in _train_dataset.targets]

        elif opt.data_order == "original":
            print("data order random")
        
        for tc in target_classes:
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
            tc_num = (np.array(_train_dataset.targets) == tc).sum()
            subset_importance_weight += list(np.ones(tc_num) / tc_num)

        _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)
        bsz = int(len(train_dataset) / 5)
        print("bsz: ", bsz)

        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=bsz, shuffle=True,
                            num_workers=opt.num_workers, pin_memory=True)
        
        train_loaders += [train_loader]

    return train_loaders


# 現在タスクのデータ+リプレイバッファのデータを含むデータローダーを返す（基本的に訓練用データローダーを同じ）
def set_gradreplay_loader_cclis_cifar100(opt, normalize, replay_indices, method_tools, training=True):

    importance_weight = method_tools['importance_weight']

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print('target_classes', target_classes)

    # 
    subset_indices = []
    subset_importance_weight = []

    _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                       train=training,
                                       transform=train_transform,
                                       download=True)
    
    # cifar100データセットのクラスを入れ替え
    if opt.data_order == "sparse2coarse":

        # cifar100のsuperclassに基づいて，クラスラベルを書き換える
        # 水性哺乳類に含まれるクラス4, 30, 55, 72, 95を0, 1, 2, 3, 4に書き換える
        # 魚に含まれるクラス1, 32, 67, 73, 91を5, 6, 7, 8, 9に書き換える
        # 残りも同様に，，，，
        remap = {}
        for g_idx, fine_list in enumerate(groups):
            base = g_idx * 5
            for offset, old_label in enumerate(fine_list):
                remap[old_label] = base + offset
        
        # _train_dataset.targets を新ラベルに書き換え
        _train_dataset.targets = [remap[int(t)] for t in _train_dataset.targets]

    elif opt.data_order == "original":
        print("data order random")
    
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
        tc_num = (np.array(_train_dataset.targets) == tc).sum()
        
        subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list

    _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

    if len(replay_indices) > 0 and training:
        prev_dataset = IS_Subset(_train_dataset, replay_indices, importance_weight)
        cur_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        train_dataset = ConcatDataset([prev_dataset, cur_dataset])

    else:
        _subset_indices += replay_indices
        _subset_importance_weight += importance_weight

        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

    subset_indices += replay_indices
    subset_importance_weight += importance_weight

    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
    replay_sample_num = uc[np.argsort(uk)]


    if len(replay_indices) > 0 and training: 
        train_batch_size_list = [int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list))), 
                                 opt.batch_size - int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list)))]
        
        print('train_batch_size', train_batch_size_list)
        train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=train_batch_size_list)
        print('len_data', [len(cur_dataset) for cur_dataset in train_sampler.dataset.datasets])
    else:
        train_sampler = None
        
    if training:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=500, shuffle=(train_sampler is None),
                            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)


    else:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=500, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True)
        print('no separate sampler')
    
    return train_loader
















# 訓練用tiny-imagenet
def set_loader_cclis_tinyimagenet(opt, normalize, replay_indices, method_tools, training=True):

    importance_weight = method_tools['importance_weight']

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print('target_classes', target_classes)

    subset_indices = []
    subset_importance_weight = []
    _train_dataset = TinyImagenet(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)
    for tc in target_classes:
        target_class_indices = np.where(_train_dataset.targets == tc)[0]
        subset_indices += np.where(_train_dataset.targets == tc)[0].tolist()
        tc_num = (np.array(_train_dataset.targets) == tc).sum()
        
        subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list

    _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

    if len(replay_indices) > 0 and training:
        prev_dataset = IS_Subset(_train_dataset, replay_indices, importance_weight)
        cur_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        train_dataset = ConcatDataset([prev_dataset, cur_dataset])

    else:
        _subset_indices += replay_indices
        _subset_importance_weight += importance_weight
        print('_subset_indices length', len(_subset_indices))
        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

    subset_indices += replay_indices
    subset_importance_weight += importance_weight

    print('dataset length', len(_train_dataset), len(train_dataset))
    print('Dataset size: {}'.format(len(subset_indices)))

    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
    replay_sample_num = uc[np.argsort(uk)]

    if len(replay_indices) > 0 and training: 
        train_batch_size_list = [int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list))), 
                                 opt.batch_size - int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list)))]
        
        print('train_batch_size', train_batch_size_list)
        train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=train_batch_size_list)
        print('len_data', [len(cur_dataset) for cur_dataset in train_sampler.dataset.datasets])
    else:
        train_sampler = None
        
    if training:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)


    else:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True)
        print('no separate sampler')
    
    return train_loader, subset_indices, replay_sample_num



# ある１クラスのサンプルのみを含む検証用データローダーの作成tiny-imagenet
def set_grad_loader_cclis_tinyimagenet(opt, train, normalize, method_tools):

    importance_weight = method_tools['importance_weight']

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    train_loaders = []

    for tc in range(opt.n_cls):

        subset_indices = []
        subset_importance_weight = []

        _train_dataset = TinyImagenet(root=opt.data_folder,
                                      train=train,
                                      transform=train_transform,
                                      download=True)
        
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]

        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
        tc_num = (np.array(_train_dataset.targets) == tc).sum()
        
        subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list

        _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)
        bsz = int(len(train_dataset) / 5)
        print("bsz: ", bsz)

        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=bsz, shuffle=True,
                            num_workers=opt.num_workers, pin_memory=True)
        
        train_loaders += [train_loader]

    return train_loaders



# ある１クラスのサンプルのみを含む検証用データローダーの作成tiny-imagenet
def set_gradtask_loader_cclis_tinyimagenet(opt, train, normalize, method_tools):

    importance_weight = method_tools['importance_weight']

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    train_loaders = []

    for task_id in range(opt.n_task):

        target_classes = list(range(task_id*opt.cls_per_task, (task_id+1)*opt.cls_per_task))

        subset_indices = []
        subset_importance_weight = []

        _train_dataset = TinyImagenet(root=opt.data_folder,
                                      train=train,
                                      transform=train_transform,
                                      download=True)
        
        for tc in target_classes:
            subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
            tc_num = (np.array(_train_dataset.targets) == tc).sum()
            subset_importance_weight += list(np.ones(tc_num) / tc_num)

        _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)
        bsz = int(len(train_dataset) / 5)
        print("bsz: ", bsz)

        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=bsz, shuffle=True,
                            num_workers=opt.num_workers, pin_memory=True)
        
        train_loaders += [train_loader]

    return train_loaders



# 現在タスクのデータ+リプレイバッファのデータを含むデータローダーを返す（基本的に訓練用データローダーを同じ）
def set_gradreplay_loader_cclis_tinyimagenet(opt, normalize, replay_indices, method_tools, training=True):

    importance_weight = method_tools['importance_weight']

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1 if opt.dataset=='tiny-imagenet' else 0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print('target_classes', target_classes)

    subset_indices = []
    subset_importance_weight = []
    _train_dataset = TinyImagenet(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)
    for tc in target_classes:
        target_class_indices = np.where(_train_dataset.targets == tc)[0]
        subset_indices += np.where(_train_dataset.targets == tc)[0].tolist()
        tc_num = (np.array(_train_dataset.targets) == tc).sum()
        
        subset_importance_weight += list(np.ones(tc_num) / tc_num)  # cur_sample importance weight, list

    _subset_indices, _subset_importance_weight = copy.deepcopy(subset_indices), copy.deepcopy(subset_importance_weight)

    if len(replay_indices) > 0 and training:
        prev_dataset = IS_Subset(_train_dataset, replay_indices, importance_weight)
        cur_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        train_dataset = ConcatDataset([prev_dataset, cur_dataset])

    else:
        _subset_indices += replay_indices
        _subset_importance_weight += importance_weight
        print('_subset_indices length', len(_subset_indices))
        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

    subset_indices += replay_indices
    subset_importance_weight += importance_weight

    print('dataset length', len(_train_dataset), len(train_dataset))
    print('Dataset size: {}'.format(len(subset_indices)))

    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
    replay_sample_num = uc[np.argsort(uk)]

    if len(replay_indices) > 0 and training: 
        train_batch_size_list = [int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list))), 
                                 opt.batch_size - int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list)))]
        
        print('train_batch_size', train_batch_size_list)
        train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=train_batch_size_list)
        print('len_data', [len(cur_dataset) for cur_dataset in train_sampler.dataset.datasets])
    else:
        train_sampler = None
        
    if training:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=500, shuffle=(train_sampler is None),
                            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)


    else:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=500, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True)
        print('no separate sampler')
    
    return train_loader, subset_indices, replay_sample_num






