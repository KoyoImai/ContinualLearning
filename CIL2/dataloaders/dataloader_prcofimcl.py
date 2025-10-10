


import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler


from dataloaders.tiny_imagenets import TinyImagenet





# EFM計算用 cifar10
def set_linearloader_efm_cifar10(opt, normalize, replay_indices):


    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        transform=train_transform,
                                        download=True)

    _train_targets = np.array(_train_dataset.targets)
    for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()


    if isinstance(replay_indices, list):
        subset_indices += replay_indices
    elif isinstance(replay_indices, np.ndarray):
        subset_indices += replay_indices.tolist()
    else:
        assert False


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








# EFM計算用 cifar100
def set_linearloader_efm_cifar100(opt, normalize, replay_indices):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                       transform=train_transform,
                                       download=True)
    
    _train_targets = np.array(_train_dataset.targets)
    for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()


    if isinstance(replay_indices, list):
        subset_indices += replay_indices
    elif isinstance(replay_indices, np.ndarray):
        subset_indices += replay_indices.tolist()
    else:
        assert False


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





# EFM計算用tiny-imagenet
def set_linearloader_efm_tinyimagenet(opt, normalize, replay_indices):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    target_classes = list(range(0, (opt.target_task+1)*opt.cls_per_task))

    subset_indices = []
    _train_dataset = TinyImagenet(root=opt.data_folder,
                                  transform=train_transform,
                                  download=True)
    _train_targets = np.array(_train_dataset.targets)
    for tc in range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task):
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
    

    if isinstance(replay_indices, list):
        subset_indices += replay_indices
    elif isinstance(replay_indices, np.ndarray):
        subset_indices += replay_indices.tolist()
    else:
        assert False

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









