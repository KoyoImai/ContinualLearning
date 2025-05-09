
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
def set_loader_cclis_wo_ss_cifar10(opt, normalize, replay_indices, method_tools, training=True):

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
        # print("importance_weight: ", importance_weight)
        _subset_indices += replay_indices
        _subset_importance_weight += importance_weight

        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

    subset_indices += replay_indices
    subset_importance_weight += importance_weight

    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
    replay_sample_num = uc[np.argsort(uk)]


    # if len(replay_indices) > 0 and training: 
    #     train_batch_size_list = [int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list))), 
    #                              opt.batch_size - int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list)))]
        
    #     print('train_batch_size', train_batch_size_list)
    #     train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=train_batch_size_list)
    #     print('len_data', [len(cur_dataset) for cur_dataset in train_sampler.dataset.datasets])
    # else:
    #     train_sampler = None

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






# 訓練用cifar100
def set_loader_cclis_wo_ss_cifar100(opt, normalize, replay_indices, method_tools, training=True):

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

    _train_dataset = datasets.CIFAR100(root=opt.data_folder,
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
        print('_subset_indices length', len(_subset_indices))
        train_dataset = IS_Subset(_train_dataset, _subset_indices, _subset_importance_weight)

    subset_indices += replay_indices
    subset_importance_weight += importance_weight

    print('dataset length', len(_train_dataset), len(train_dataset))        
    print('Dataset size: {}'.format(len(subset_indices)))

    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
    replay_sample_num = uc[np.argsort(uk)]

    # if len(replay_indices) > 0 and training: 
    #     train_batch_size_list = [int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list))), 
    #                              opt.batch_size - int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list)))]
        
    #     print('train_batch_size', train_batch_size_list)
    #     train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=train_batch_size_list)
    #     print('len_data', [len(cur_dataset) for cur_dataset in train_sampler.dataset.datasets])
    # else:
    #     train_sampler = None

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






# 訓練用tiny-imagenet
def set_loader_cclis_wo_ss_tinyimagenet(opt, normalize, replay_indices, method_tools, training=True):

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

    # if len(replay_indices) > 0 and training: 
    #     train_batch_size_list = [int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list))), 
    #                              opt.batch_size - int(np.round(opt.batch_size * dataset_len_list[0] / sum(dataset_len_list)))]
        
    #     print('train_batch_size', train_batch_size_list)
    #     train_sampler = BatchSchedulerSampler(dataset=train_dataset, batch_size=train_batch_size_list)
    #     print('len_data', [len(cur_dataset) for cur_dataset in train_sampler.dataset.datasets])
    # else:
    #     train_sampler = None

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



