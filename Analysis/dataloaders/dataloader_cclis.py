


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



# CIFAR10
def set_loader_cclis_cifar10(opt, normalize, replay_indices=np.array([])):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクまでの全てのクラスを対象
    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)

    _train_targets = np.array(_train_dataset.targets)
    # for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
    for tc in range(0, (opt.target_task+1)*opt.cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

    subset_indices += replay_indices.tolist()

    ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)

    train_dataset =  Subset(_train_dataset, subset_indices)

    subset_indices = []
    _val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                    train=False,
                                    transform=train_transform)
    for tc in target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500,
        shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=500,
        shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    
    return train_loader, val_loader


def set_loader_cclis_cifar10_v2(opt, normalize, replay_indices=[]):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクまでの全てのクラスを対象
    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)

    _train_targets = np.array(_train_dataset.targets)
    for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

    subset_indices += replay_indices.tolist()

    ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)

    train_dataset =  Subset(_train_dataset, subset_indices)

    subset_indices = []
    _val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                    train=False,
                                    transform=train_transform)
    for tc in target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500,
        shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=500,
        shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    
    return train_loader, val_loader



def set_taskil_valloader_cclis_cifar10(opt, normalize):

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    val_loaders = []

    for task_id in range(opt.n_task):
        target_classes = list(range(task_id*opt.cls_per_task, (task_id+1)*opt.cls_per_task))

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
        
        val_loaders += [val_loader]

    return val_loaders








# CIFAR100
def set_loader_cclis_cifar100(opt, normalize, replay_indices=np.array([])):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクまでの全てのクラスを対象
    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)

    _train_targets = np.array(_train_dataset.targets)
    # for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
    for tc in range(0, (opt.target_task+1)*opt.cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

    subset_indices += replay_indices.tolist()

    ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)

    train_dataset =  Subset(_train_dataset, subset_indices)

    subset_indices = []
    _val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                    train=False,
                                    transform=train_transform)
    for tc in target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500,
        shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=500,
        shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    
    return train_loader, val_loader


def set_loader_cclis_cifar100_v2(opt, normalize, replay_indices=[]):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクまでの全てのクラスを対象
    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)

    _train_targets = np.array(_train_dataset.targets)
    for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

    subset_indices += replay_indices.tolist()

    ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)

    train_dataset =  Subset(_train_dataset, subset_indices)

    subset_indices = []
    _val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                    train=False,
                                    transform=train_transform)
    for tc in target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500,
        shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=500,
        shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    
    return train_loader, val_loader


def set_taskil_valloader_cclis_cifar100(opt, normalize):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    val_loaders = []

    for task_id in range(opt.n_task):

        target_classes = list(range(task_id*opt.cls_per_task, (task_id+1)*opt.cls_per_task))

        subset_indices = []
        _val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         train=False,
                                         transform=train_transform)
        
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=500, shuffle=None,
            num_workers=opt.num_workers, pin_memory=True)
        
        val_loaders += [val_loader]

    return val_loaders






# tiny-imagenet
def set_loader_cclis_tinyimagenet(opt, normalize, replay_indices=np.array([])):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクまでの全てのクラスを対象
    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _train_dataset = TinyImagenet(root=opt.data_folder,
                                  transform=train_transform,
                                  download=True)

    _train_targets = np.array(_train_dataset.targets)
    # for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
    for tc in range(0, (opt.target_task+1)*opt.cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

    subset_indices += replay_indices.tolist()

    ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)

    train_dataset =  Subset(_train_dataset, subset_indices)

    subset_indices = []
    _val_dataset = TinyImagenet(root=opt.data_folder,
                                train=False,
                                transform=train_transform)
    for tc in target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500,
        shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=500,
        shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    
    return train_loader, val_loader




def set_loader_cclis_tinyimagenet_v2(opt, normalize, replay_indices=[]):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクまでの全てのクラスを対象
    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _train_dataset = TinyImagenet(root=opt.data_folder,
                                  transform=train_transform,
                                  download=True)

    _train_targets = np.array(_train_dataset.targets)
    for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

    subset_indices += replay_indices.tolist()

    ut, uc = np.unique(_train_targets[subset_indices], return_counts=True)
    print(ut)
    print(uc)

    train_dataset =  Subset(_train_dataset, subset_indices)

    subset_indices = []
    _val_dataset = TinyImagenet(root=opt.data_folder,
                                train=False,
                                transform=train_transform)
    for tc in target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500,
        shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=500,
        shuffle=False, num_workers=opt.num_workers, pin_memory=True)
    
    return train_loader, val_loader



def set_taskil_valloader_cclis_tinyimagenet(opt, normalize):

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    val_loaders = []

    for task_id in range(opt.n_task):
        target_classes = list(range(task_id*opt.cls_per_task, (task_id+1)*opt.cls_per_task))

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
        
        val_loaders += [val_loader]

    return val_loaders





