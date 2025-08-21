
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
    


def co2l_dataloader_cifar10(opt, normalize, train):

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
    target_classes = opt.cls_list
    print(target_classes)

    subset_indices = []
    _train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                      transform=TwoCropTransform(train_transform),
                                      download=True)
    
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()
    
    train_dataset = Subset(_train_dataset, subset_indices)


    print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    return train_loader









