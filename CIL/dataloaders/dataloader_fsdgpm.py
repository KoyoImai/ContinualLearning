import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset

from dataloaders.tiny_imagenets import TinyImagenet



# 訓練用CIFAR10
def set_loader_fsdgpm_cifar10(opt, normalize, replay_indices):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクのクラス
    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    subset_indices = []
    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

    print("replay_indices: ", replay_indices)
    subset_indices += replay_indices

    train_dataset =  Subset(_train_dataset, subset_indices)
    print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader, subset_indices


# 検証用cifar10
def set_valloader_fsdgpm_cifar10(opt, normalize):

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


# vanilla用cifar10
def set_vanillaloader_fsdgpm_cifar10(opt, normalize):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクのクラス
    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    subset_indices = []
    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        transform=train_transform,
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
        train_dataset, batch_size=500, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, subset_indices



# 訓練用cifar100
def set_loader_fsdgpm_cifar100(opt, normalize, replay_indices):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.1, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクのクラス
    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    subset_indices = []
    _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

    subset_indices += replay_indices

    train_dataset =  Subset(_train_dataset, subset_indices)
    print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])


    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader, subset_indices


# 検証用cifar100
def set_valloader_fsdgpm_cifar100(opt, normalize):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        # transforms.RandomResizedCrop(size=opt.size, scale=(0.1, 1.)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                       train=False,
                                       transform=train_transform)
    
    for tc in target_classes:
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return val_loader


# vanilla用cifar100
def set_vanillaloader_fsdgpm_cifar100(opt, normalize, replay_indices):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクのクラス
    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    subset_indices = []
    _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        transform=train_transform,
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
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, subset_indices


# 訓練用tiny-imagenet
def set_loader_fsdgpm_tinyimagenet(opt, normalize, replay_indices):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクのクラス
    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    subset_indices = []

    _train_dataset = TinyImagenet(root=opt.data_folder,
                                  transform=train_transform,
                                  download=True)
    for tc in target_classes:
        target_class_indices = np.where(_train_dataset.targets == tc)[0]
        subset_indices += np.where(_train_dataset.targets == tc)[0].tolist()

    subset_indices += replay_indices

    train_dataset =  Subset(_train_dataset, subset_indices)
    print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader, subset_indices


# 検証用tiny-imagenet
def set_valloader_fsdgpm_tinyimagenet(opt, normalize):

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


# vanilla用tiny-imagenet
def set_vanillaloader_fsdgpm_tinyimagenet(opt, normalize):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクのクラス
    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    subset_indices = []

    _train_dataset = TinyImagenet(root=opt.data_folder,
                                  transform=train_transform,
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






















