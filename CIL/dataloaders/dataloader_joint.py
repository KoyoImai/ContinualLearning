
import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler

from dataloaders.tiny_imagenets import TinyImagenet



class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]





# 訓練用cifar10
def set_loader_joint_cifar10(opt, normalize):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    # これまでのタスクのクラス
    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    subset_indices = []
    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                      transform=TwoCropTransform(train_transform),
                                      download=True)
    
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()


    train_dataset =  Subset(_train_dataset, subset_indices)
    print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader, subset_indices


# 線形層訓練用cifar10
def set_linearloader_joint_cifar10(opt, normalize):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)

    _train_targets = np.array(_train_dataset.targets)
    for tc in range(0, (opt.target_task+1)*opt.cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()


    ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)

    weights = np.array([0.] * len(subset_indices))
    for t, c in zip(ut, uc):
        weights[_train_targets[subset_indices] == t] = 1./c

    train_dataset =  Subset(_train_dataset, subset_indices)

    train_sampler = WeightedRandomSampler(torch.Tensor(weights), len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.linear_batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    
    return train_loader


# 検証用cifar10
def set_valloader_joint_cifar10(opt, normalize):

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                    train=False,
                                    transform=val_transform)
    for tc in target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return val_loader






# 訓練用cifar100
def set_loader_joint_cifar100(opt, normalize):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクのクラス
    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    subset_indices = []
    _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                       transform=TwoCropTransform(train_transform),
                                       download=True)
    
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()


    train_dataset =  Subset(_train_dataset, subset_indices)
    print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader, subset_indices


# 線形層訓練用cifar100
def set_linearloader_joint_cifar100(opt, normalize):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                       transform=train_transform,
                                       download=True)
    
    _train_targets = np.array(_train_dataset.targets)
    for tc in range(0, (opt.target_task+1)*opt.cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()


    ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)

    weights = np.array([0.] * len(subset_indices))
    for t, c in zip(ut, uc):
        weights[_train_targets[subset_indices] == t] = 1./c

    train_dataset =  Subset(_train_dataset, subset_indices)

    train_sampler = WeightedRandomSampler(torch.Tensor(weights), len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.linear_batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    
    return train_loader


# 検証用cifar100
def set_valloader_joint_cifar100(opt, normalize):

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                    train=False,
                                    transform=val_transform)
    
    for tc in target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return val_loader






# 訓練用tiny-imagenet
def set_loader_joint_tinyimagenet(opt, normalize):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクのクラス
    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    subset_indices = []

    _train_dataset = TinyImagenet(root=opt.data_folder,
                                  transform=TwoCropTransform(train_transform),
                                  download=True)
    
    for tc in target_classes:
        target_class_indices = np.where(_train_dataset.targets == tc)[0]
        subset_indices += np.where(_train_dataset.targets == tc)[0].tolist()


    train_dataset =  Subset(_train_dataset, subset_indices)
    print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, subset_indices


# 線形層学習用tiny-imagenet
def set_linearloader_joint_tinyimagenet(opt, normalize):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _train_dataset = TinyImagenet(root=opt.data_folder,
                                  transform=train_transform,
                                  download=True)
    _train_targets = np.array(_train_dataset.targets)
    for tc in range(0, (opt.target_task+1)*opt.cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
    

    ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)

    weights = np.array([0.] * len(subset_indices))
    for t, c in zip(ut, uc):
        weights[_train_targets[subset_indices] == t] = 1./c

    train_dataset =  Subset(_train_dataset, subset_indices)

    train_sampler = WeightedRandomSampler(torch.Tensor(weights), len(weights))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.linear_batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    
    return train_loader


# 検証用tiny-imagenet
def set_valloader_joint_tinyimagenet(opt, normalize):

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _val_dataset = TinyImagenet(root=opt.data_folder,
                                    train=False,
                                    transform=val_transform)
    for tc in target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return val_loader




