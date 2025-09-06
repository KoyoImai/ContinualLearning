import copy
import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data import Subset, Dataset
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import Sampler, RandomSampler


from dataloaders.tiny_imagenets import TinyImagenet




class ER_Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        index = self.indices[idx]
        image, label = self.dataset[index]

        # print("[DEBUG] __getitem__ called")  # ← 絶対呼ばれるはず
        return image, label, index

    def __len__(self):
        return len(self.indices)



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

    # def __len__(self):
    #     return self.dataset_len

    # def __iter__(self):
    #     samplers_list = []
    #     sampler_iterators = []
    #     for dataset_idx in range(self.number_of_datasets):
    #         cur_dataset = self.dataset.datasets[dataset_idx]
    #         sampler = RandomSampler(cur_dataset) 
    #         samplers_list.append(sampler)
    #         cur_sampler_iterator = sampler.__iter__()
    #         sampler_iterators.append(cur_sampler_iterator)

    #     push_index_val = [0] + self.dataset.cumulative_sizes[:-1] 
    #     step = sum(self.batch_size) 

    #     samples_to_grab, epoch_samples = self.batch_size, self.dataset_len  
    #     # print('epoch_samples', epoch_samples)

    #     final_samples_list = []  # this is a list of indexes from the combined dataset
    #     for _ in range(0, epoch_samples, step):
    #         for i in range(self.number_of_datasets):
    #             cur_batch_sampler = sampler_iterators[i]
    #             cur_samples = []
    #             for _ in range(samples_to_grab[i]):
    #                 try:
    #                     cur_sample_org = cur_batch_sampler.__next__()
    #                     cur_sample = cur_sample_org + push_index_val[i]
    #                     cur_samples.append(cur_sample)
    #                 except StopIteration: 
    #                     # got to the end of iterator - restart the iterator and continue to get samples
    #                     # until reaching "epoch_samples"
    #                     break

    #             final_samples_list.extend(cur_samples)

    #     return iter(final_samples_list)

    def __len__(self):
        step = sum(self.batch_size)
        return (self.dataset_len // step) * step  # フルバッチ単位に揃える

    def __iter__(self):
        from torch.utils.data import RandomSampler
        # 各サブデータセットごとにサンプラとイテレータ
        samplers = [RandomSampler(ds) for ds in self.dataset.datasets]
        iters = [iter(s) for s in samplers]

        push = [0] + self.dataset.cumulative_sizes[:-1]   # ConcatDataset用オフセット
        step = sum(self.batch_size)
        epoch_samples = len(self)                         # フルバッチ×step

        out = []
        for _ in range(0, epoch_samples, step):
            for i, need in enumerate(self.batch_size):
                cur = []
                while len(cur) < need:
                    try:
                        j_local = next(iters[i])
                    except StopIteration:
                        # 枯渇したら即リスタート（枚数をきっちり満たす）
                        samplers[i] = RandomSampler(self.dataset.datasets[i])
                        iters[i] = iter(samplers[i])
                        j_local = next(iters[i])
                    cur.append(j_local + push[i])
                out.extend(cur)
        return iter(out)






# 訓練用CIFAR10
def set_loader_er_cifar10(opt, normalize, replay_indices, training=True):

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

    _subset_indices = copy.deepcopy(subset_indices)

    if len(replay_indices) > 0 and training: 
        prev_dataset = ER_Subset(_train_dataset, replay_indices)
        cur_dataset = ER_Subset(_train_dataset, _subset_indices)

        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        train_dataset = ConcatDataset([prev_dataset, cur_dataset])
    
    else:
        _subset_indices += replay_indices

        train_dataset = ER_Subset(_train_dataset, _subset_indices)

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
    

    # # Samplerが正しく機能しているかの確認
    # for b, (imgs, labels, idxs) in enumerate(train_loader):

    #     # 2) クラス別枚数
    #     uniq, cnt = labels.unique(return_counts=True)
    #     print("  class_counts:", dict(zip(uniq.tolist(), cnt.tolist())))


    return train_loader, subset_indices


# 検証用cifar10
def set_valloader_er_cifar10(opt, normalize):

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
def set_vanillaloader_er_cifar10(opt, normalize):

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
    # print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, subset_indices


# NCM分類用cifar10
def set_ncmloader_er_cifar10(opt, normalize, replay_indices):

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

    # print("replay_indices: ", replay_indices)
    subset_indices += replay_indices

    train_dataset =  Subset(_train_dataset, subset_indices)
    # print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, subset_indices


# task-il 検証用cifar10
def set_taskil_valloader_er_cifar10(opt, normalize, train=False):

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
                                        train=train,
                                        transform=val_transform)
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=8, pin_memory=True)
        
        val_loaders += [val_loader]

    return val_loaders


# ある１クラスのサンプルのみを含む検証用データローダーの作成cifar10
def set_gard_loader_er_cifar10(opt, train, normalize):

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    val_loaders = []

    for tc in range(opt.n_cls):

        subset_indices = []
        _val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                        train=train,
                                        transform=val_transform)

        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)
        bsz = len(val_dataset)
        print("vak_dataset size: ", len(val_dataset))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=bsz, shuffle=False,
            num_workers=8, pin_memory=True)
        

        
        val_loaders += [val_loader]

    return val_loaders


# ある１タスクのサンプルのみを含む検証用データローダーの作成cifar10
def set_gardtask_loader_er_cifar10(opt, train, normalize):

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
                                        train=train,
                                        transform=val_transform)

        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()

        val_dataset =  Subset(_val_dataset, subset_indices)
        bsz = len(val_dataset)
        print("vak_dataset size: ", len(val_dataset))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=bsz, shuffle=False,
            num_workers=8, pin_memory=True)
        
        val_loaders += [val_loader]

    return val_loaders


# 全てのサンプルを含んだデータローダーを返す
def set_grad_loader_er_cifar10_v2(opt, train, normalize, replay_indices=None):

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])


    subset_indices = []

    _val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                    train=train,
                                    transform= val_transform)

    subset_indices += np.array(_val_dataset.targets).tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)
    
    bsz = int(len(val_dataset) / 100)
    print("vak_dataset size: ", len(val_dataset))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bsz, shuffle=True,
        num_workers=8, pin_memory=True)

    return val_loader


# 現在タスクのデータ+リプレイバッファのデータを含むデータローダーを返す（基本的に訓練用データローダーを同じ）
def set_gradreplay_loader_er_cifar10(opt, train, normalize, replay_indices=[], training=True):

    val_transform = transforms.Compose([
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
                                      train=train,
                                      transform= val_transform,
                                      download=True)
    
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

    _subset_indices = copy.deepcopy(subset_indices)

    if len(replay_indices) > 0 and training: 
        prev_dataset = ER_Subset(_train_dataset, replay_indices)
        cur_dataset = ER_Subset(_train_dataset, _subset_indices)

        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        train_dataset = ConcatDataset([prev_dataset, cur_dataset])
    
    else:
        _subset_indices += replay_indices

        train_dataset = ER_Subset(_train_dataset, _subset_indices)

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
                            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=False)


    else:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    

    # # Samplerが正しく機能しているかの確認
    # for b, (imgs, labels, idxs) in enumerate(train_loader):

    #     # 2) クラス別枚数
    #     uniq, cnt = labels.unique(return_counts=True)
    #     print("  class_counts:", dict(zip(uniq.tolist(), cnt.tolist())))


    return train_loader
















# 訓練用cifar100
def set_loader_er_cifar100(opt, normalize, replay_indices, training=True):

    # print("replay_indices: ", replay_indices)

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
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()  # cur_sample index, list
    
    _subset_indices = copy.deepcopy(subset_indices)

    if len(replay_indices) > 0 and training:
        prev_dataset = ER_Subset(_train_dataset, replay_indices)
        cur_dataset = ER_Subset(_train_dataset, _subset_indices)

        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        train_dataset = ConcatDataset([prev_dataset, cur_dataset])
    else:
        _subset_indices += replay_indices
        print('_subset_indices length', len(_subset_indices))
        train_dataset = ER_Subset(_train_dataset, _subset_indices)
    
    subset_indices += replay_indices

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
    

    # Samplerが正しく機能しているかの確認
    for b, (imgs, labels, idxs) in enumerate(train_loader):

        # 2) クラス別枚数
        uniq, cnt = labels.unique(return_counts=True)
        print("  class_counts:", dict(zip(uniq.tolist(), cnt.tolist())))
    
    return train_loader, subset_indices


# 検証用cifar100
def set_valloader_er_cifar100(opt, normalize):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
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
        val_dataset, batch_size=500, shuffle=None,
        num_workers=opt.num_workers, pin_memory=True)

    return val_loader


# vanilla用cifar100
def set_vanillaloader_er_cifar100(opt, normalize):

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
    # print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])


    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, subset_indices


# NCM分類用cifar100
def set_ncmloader_er_cifar100(opt, normalize, replay_indices):

    # print("replay_indices: ", replay_indices)

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

    subset_indices += replay_indices

    train_dataset =  Subset(_train_dataset, subset_indices)
    # print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])


    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, subset_indices


# 検証用cifar100
def set_taskil_valloader_er_cifar100(opt, normalize, train=False):

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
                                         train=train,
                                         transform=train_transform)
        
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=500, shuffle=None,
            num_workers=opt.num_workers, pin_memory=True)
        
        val_loaders += [val_loader]

    return val_loaders


# ある１クラスのサンプルのみを含む検証用データローダーの作成cifar100
def set_gard_loader_er_cifar100(opt, train, normalize):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    val_loaders = []

    for tc in range(opt.n_cls):

        subset_indices = []
        _val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         train=train,
                                         transform=train_transform)
        
        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)
        bsz = len(val_dataset)
        print("vak_dataset size: ", len(val_dataset))

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=bsz, shuffle=None,
            num_workers=opt.num_workers, pin_memory=True)
        
        val_loaders += [val_loader]

    return val_loaders


# ある１タスクのサンプルのみを含む検証用データローダーの作成cifar100
def set_gardtask_loader_er_cifar100(opt, train, normalize):

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    val_loaders = []

    for task_id in range(opt.n_task):
        target_classes = list(range(task_id*opt.cls_per_task, (task_id+1)*opt.cls_per_task))

        subset_indices = []
        _val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                         train=train,
                                         transform=val_transform)

        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()

        val_dataset =  Subset(_val_dataset, subset_indices)
        bsz = len(val_dataset)
        print("vak_dataset size: ", len(val_dataset))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=bsz, shuffle=False,
            num_workers=8, pin_memory=True)
        
        val_loaders += [val_loader]

    return val_loaders


# 全てのサンプルを含んだデータローダーを返す
def set_grad_loader_er_cifar100_v2(opt, train, normalize, replay_indices=None):

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    subset_indices = []

    _val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                     train=train,
                                     transform=val_transform)

    subset_indices += np.array(_val_dataset.targets).tolist()
    val_dataset =  Subset(_val_dataset, subset_indices)
    
    bsz = int(len(val_dataset) / 100)
    print("vak_dataset size: ", len(val_dataset))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bsz, shuffle=True,
        num_workers=8, pin_memory=True)

    return val_loader


# 現在タスクのデータ+リプレイバッファのデータを含むデータローダーを返す（基本的に訓練用データローダーを同じ）
def set_gradreplay_loader_er_cifar100(opt, train, normalize, replay_indices=[], training=True):

    val_transform = transforms.Compose([
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
                                       train=train,
                                       transform=val_transform,
                                       download=True)
    
    for tc in target_classes:
        target_class_indices = np.where(np.array(_train_dataset.targets) == tc)[0]
        subset_indices += np.where(np.array(_train_dataset.targets) == tc)[0].tolist()

    _subset_indices = copy.deepcopy(subset_indices)

    if len(replay_indices) > 0 and training:
        prev_dataset = ER_Subset(_train_dataset, replay_indices)
        cur_dataset = ER_Subset(_train_dataset, _subset_indices)

        dataset_len_list = [len(prev_dataset), len(cur_dataset)]

        train_dataset = ConcatDataset([prev_dataset, cur_dataset])
    else:
        _subset_indices += replay_indices
        print('_subset_indices length', len(_subset_indices))
        train_dataset = ER_Subset(_train_dataset, _subset_indices)
    
    subset_indices += replay_indices

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
                            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=False)
    else:
        train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
        print('no separate sampler')
    

    # # Samplerが正しく機能しているかの確認
    # for b, (imgs, labels, idxs) in enumerate(train_loader):

    #     # 2) クラス別枚数
    #     uniq, cnt = labels.unique(return_counts=True)
    #     print("  class_counts:", dict(zip(uniq.tolist(), cnt.tolist())))
    
    return train_loader
















# 訓練用tiny-imagenet
def set_loader_er_tinyimagenet(opt, normalize, replay_indices):

    train_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomApply([
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        # ], p=0.8),
        # transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=opt.size//20*2+1, sigma=(0.1, 2.0))], p=0.5 if opt.size>32 else 0.0),
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
    # print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, subset_indices


# 検証用tiny-imagenet
def set_valloader_er_tinyimagenet(opt, normalize):

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
def set_vanillaloader_er_tinyimagenet(opt, normalize):

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
    # print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.vanilla_batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, subset_indices


# NCM分類用tiny-imagenet
def set_ncmloader_er_tinyimagenet(opt, normalize, replay_indices):

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

    subset_indices += replay_indices

    train_dataset =  Subset(_train_dataset, subset_indices)
    # print('Dataset size: {}'.format(len(subset_indices)))
    uk, uc = np.unique(np.array(_train_dataset.targets)[subset_indices], return_counts=True)
    print(uc[np.argsort(uk)])

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=500, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, subset_indices


# taskil 検証用tiny-imagenet
def set_taskil_valloader_er_tinyimagenet(opt, normalize, train=False):

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
                                        train=train,
                                        transform=val_transform)
        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=256, shuffle=False,
            num_workers=8, pin_memory=True)
        
        val_loaders += [val_loader]

    return val_loaders


# ある１クラスのサンプルのみを含む検証用データローダーの作成tiny-imagenet
def set_grad_loader_er_tinyimagenet(opt, train, normalize):

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    val_loaders = []

    for tc in range(opt.n_cls):

        subset_indices = []
        _val_dataset = TinyImagenet(root=opt.data_folder,
                                        train=train,
                                        transform=val_transform)

        subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()
        val_dataset =  Subset(_val_dataset, subset_indices)
        bsz = len(val_dataset)
        print("vak_dataset size: ", len(val_dataset))

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=bsz, shuffle=False,
            num_workers=8, pin_memory=True)
        
        val_loaders += [val_loader]

    return val_loaders


# ある１タスクのサンプルのみを含む検証用データローダーの作成tiny-imagenet
def set_gardtask_loader_er_tinyimagenet(opt, train, normalize):

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
                                        train=train,
                                        transform=val_transform)


        for tc in target_classes:
            subset_indices += np.where(np.array(_val_dataset.targets) == tc)[0].tolist()

        val_dataset =  Subset(_val_dataset, subset_indices)
        bsz = len(val_dataset)
        print("vak_dataset size: ", len(val_dataset))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=bsz, shuffle=False,
            num_workers=8, pin_memory=True)
        
        val_loaders += [val_loader]

    return val_loaders


# 全てのサンプルを含んだデータローダーを返す
def set_grad_loader_er_tinyimagenet_v2(opt, train, normalize):

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    subset_indices = []
    _train_dataset = TinyImagenet(root=opt.data_folder,
                                    transform=val_transform,
                                    train=train,
                                    download=True)

    subset_indices += np.array(_train_dataset.targets).tolist()
    val_dataset =  Subset(_train_dataset, subset_indices)
    bsz = int(len(val_dataset) / 100)
    print("vak_dataset size: ", len(val_dataset))

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bsz, shuffle=True,
        num_workers=8, pin_memory=True)
    

    return val_loader



def set_gradreplay_loader_er_tinyimagenet(opt, train, normalize, replay_indices=[]):

    val_transform = transforms.Compose([
        transforms.Resize(size=(opt.size, opt.size)),
        transforms.ToTensor(),
        normalize,
    ])

    # 現在タスクのクラス
    target_classes = list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task))
    print(target_classes)

    subset_indices = []

    _train_dataset = TinyImagenet(root=opt.data_folder,
                                  train=train,
                                  ttransform=val_transform,
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
        train_dataset, batch_size=500, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader, subset_indices




















