

from torchvision import transforms, datasets


from dataloaders.dataloader_cclis import set_loader_cclis_cifar10, set_loader_cclis_cifar10_v2, set_taskil_valloader_cclis_cifar10
from dataloaders.dataloader_cclis import set_loader_cclis_cifar100, set_loader_cclis_cifar100_v2, set_taskil_valloader_cclis_cifar100



def set_loader(opt, replay_indices):

    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':       # scaleから
        # mean = (0.5071, 0.4867, 0.4408)
        # std = (0.2675, 0.2565, 0.2761)
        mean=[x/255 for x in [125.3,123.0,113.9]]
        std=[x/255 for x in [63.0,62.1,66.7]]
    elif opt.dataset == 'tiny-imagenet':
        mean = (0.4802, 0.4480, 0.3975)
        std = (0.2770, 0.2691, 0.2821)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.method in ["cclis", "er", "co2l", "supcon-joint"]:
        if opt.dataset == "cifar10":
            train_loader, val_laoder = set_loader_cclis_cifar10(opt=opt, normalize=normalize)  # これまでのタスクのサンプルを全て含むデータローダー
            train_loader_2, _ = set_loader_cclis_cifar10_v2(opt=opt, normalize=normalize,
                                                            replay_indices=replay_indices)     # 現在タスクのデータとリプレイサンプルを含むデータローダー
            task_loaders = set_taskil_valloader_cclis_cifar10(opt=opt, normalize=normalize)    # タスク毎のデータローダー
        elif opt.dataset == "cifar100":
            train_loader, val_laoder = set_loader_cclis_cifar100(opt=opt, normalize=normalize)  # これまでのタスクのサンプルを全て含むデータローダー
            train_loader_2, _ = set_loader_cclis_cifar100_v2(opt=opt, normalize=normalize,
                                                            replay_indices=replay_indices)     # 現在タスクのデータとリプレイサンプルを含むデータローダー
            task_loaders = set_taskil_valloader_cclis_cifar100(opt=opt, normalize=normalize)    # タスク毎のデータローダー
        elif opt.dataset == "tiny-imagenet":
            assert False
    else:
        assert False
    
    data_loaders = {"train": train_loader, "trainv2": train_loader_2, "val": val_laoder, "task": task_loaders}

    return data_loaders