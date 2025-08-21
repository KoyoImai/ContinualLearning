
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



def cclis_dataloader_cifar10(opt, normalize, training):

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

    target_classes = opt.cls_list
    print("target_classes: ", target_classes)

    subset_indices = []
    subset_importance_weight = []

    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                      transform=train_transform,
                                      download=True,
                                      train=training)
    

    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
        tc_num = (np.array(_train_dataset.targets) == tc).sum()

        subset_importance_weight += list(np.ones(tc_num) / tc_num)
    
    train_dataset = IS_Subset(_train_dataset, subset_indices, subset_importance_weight)
    
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)  
    print('uc[np.argsort(uk)]', uc[np.argsort(uk)])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                               num_workers=8, pin_memory=True)

    return train_loader