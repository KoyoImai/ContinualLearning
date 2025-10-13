
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



class PRCO_Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        

    def __getitem__(self, idx):
        
        index = self.indices[idx]
        image, label = self.dataset[index]

        # print("[DEBUG] __getitem__ called")
        return image, label, index

    def __len__(self):
        return len(self.indices)



#===========================================================================
# 各タスク，各クラスのサンプル数を指定してミニバッチを作成するバッチサンプラ
#===========================================================================
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





#=========================
# 訓練用cifar10
#=========================
def set_replayonly_loader_cifar10(opt, normalize, replay_indices, model, training=True):

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

    # タスク "opt.target_task" で学習に使用するクラスのリスト
    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print('target_classes', target_classes)

    # データセットとして使用するサンプルのインデックスリスト
    subset_indices = []

    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)
    
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
        tc_num = (np.array(_train_dataset.targets) == tc).sum()
    
    _subset_indices = copy.deepcopy(subset_indices)


    if len(replay_indices) > 0 and training:
        prev_dataset = PRCO_Subset(_train_dataset, replay_indices)
        cur_dataset = PRCO_Subset(_train_dataset, _subset_indices)
        dataset_len_list = [len(prev_dataset), len(cur_dataset)]
        train_dataset = ConcatDataset([prev_dataset, cur_dataset])

    else:
        _subset_indices += replay_indices
        train_dataset = PRCO_Subset(_train_dataset, _subset_indices)

    subset_indices += replay_indices
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
    

    return train_loader, subset_indices, replay_sample_num




#=========================
# 訓練用cifar100
#=========================
def set_replayonly_loader_cifar100(opt, normalize, replay_indices, model, training=True):


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

    # cifar100データセット
    subset_indices = []
    _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)

    _train_targets = np.array(_train_dataset.targets)

    if isinstance(replay_indices, list):
        subset_indices += replay_indices
    elif isinstance(replay_indices, np.ndarray):
        subset_indices += replay_indices.tolist()
    else:
        assert False


    subset_indices += replay_indices
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])
    replay_sample_num = uc[np.argsort(uk)]


    ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)

    train_dataset =  Subset(_train_dataset, subset_indices)
    print("len(train_dataset): ", len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)


    return train_loader, subset_indices, replay_sample_num






#=========================
# 訓練用tiny-imagenet
#=========================
def set_replayonly_loader_tinyimagenet(opt, normalize, replay_indices, model, training=True):

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
    _train_dataset = TinyImagenet(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)
    for tc in target_classes:
        target_class_indices = np.where(_train_dataset.targets == tc)[0]
        subset_indices += np.where(_train_dataset.targets == tc)[0].tolist()
        tc_num = (np.array(_train_dataset.targets) == tc).sum()
    

    _subset_indices = copy.deepcopy(subset_indices)

    if len(replay_indices) > 0 and training:
        prev_dataset = PRCO_Subset(_train_dataset, replay_indices)
        cur_dataset = PRCO_Subset(_train_dataset, _subset_indices)
        dataset_len_list = [len(prev_dataset), len(cur_dataset)]
        train_dataset = ConcatDataset([prev_dataset, cur_dataset])
    else:
        _subset_indices += replay_indices
        train_dataset = PRCO_Subset(_train_dataset, _subset_indices)

    subset_indices += replay_indices
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
    

    return train_loader, subset_indices, replay_sample_num





