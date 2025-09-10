

from torchvision import transforms, datasets


# er
from dataloaders.dataloader_er import set_vanillaloader_er_cifar10, set_vanillaloader_er_cifar100, set_vanillaloader_er_tinyimagenet
from dataloaders.dataloader_er import set_ncmloader_er_cifar10, set_ncmloader_er_cifar100, set_ncmloader_er_tinyimagenet
from dataloaders.dataloader_er import set_taskil_valloader_er_cifar10, set_taskil_valloader_er_cifar100, set_taskil_valloader_er_tinyimagenet

# co2l
# from dataloaders.dataloader_co2l import set_loader_co2l_imagenet32    #ラベルなしデータセット
from dataloaders.dataloader_co2l import set_loader_co2l_cifar10, set_linearloader_co2l_cifar10, set_valloader_co2l_cifar10
from dataloaders.dataloader_co2l import set_loader_co2l_cifar100, set_linearloader_co2l_cifar100, set_valloader_co2l_cifar100
from dataloaders.dataloader_co2l import set_loader_co2l_tinyimagenet, set_linearloader_co2l_tinyimagenet, set_valloader_co2l_tinyimagenet

# cclis
from dataloaders.dataloader_cclis import set_loader_cclis_cifar10, set_loader_cclis_cifar100, set_loader_cclis_tinyimagenet

# prco
from dataloaders.dataloader_prco import set_loader_prco_cifar10, set_loader_prco_cifar100, set_loader_prco_tinyimagenet






# ==============================================================
# データローダーの作成
# ==============================================================
def set_loader(opt, model, replay_indices, method_tools):

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



    if opt.method in ["er"]:

        assert False
    
    elif opt.method in ["co2l"]:

        # if opt.unlabeled_dataset == "imagenet32":
        #     unlabeled_loader = set_loader_co2l_imagenet32(opt=opt, root=opt.data_folder, normalize=normalize)
        #     print("len(unlabeled_loader): ", len(unlabeled_loader))            

        if opt.dataset == "cifar10":
            train_loader, subset_indices = set_loader_co2l_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_co2l_cifar10(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
        elif opt.dataset == "cifar100":
            train_loader, subset_indices = set_loader_co2l_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_co2l_cifar100(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
        elif opt.dataset == 'tiny-imagenet':
            train_loader, subset_indices = set_loader_co2l_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            # val_loader = set_valloader_co2l_tinyimagenet(opt=opt, normalize=normalize)
            # linear_loader = set_linearloader_co2l_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = None
            linear_loader = None


    elif opt.method in ["cclis"]:

        # if opt.unlabeled_dataset == "imagenet32":
        #     unlabeled_loader = set_loader_co2l_imagenet32(opt=opt, root=opt.data_folder, normalize=normalize)

        if opt.dataset == 'cifar10':

            train_loader, subset_indices, subset_sample_num = set_loader_cclis_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices, model=model)
            post_loader, _, _ = set_loader_cclis_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices, model=model, training=False)
            val_loader = set_valloader_co2l_cifar10(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)

        elif opt.dataset == "cifar100":
            train_loader, subset_indices, subset_sample_num = set_loader_cclis_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices, model=model, training=True)
            post_loader, _, _ = set_loader_cclis_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices, model=model, training=False)
            val_loader = set_valloader_co2l_cifar100(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)

        elif opt.dataset == 'tiny-imagenet':
            train_loader, subset_indices, subset_sample_num = set_loader_cclis_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices, model=model)
            post_loader, _, _ = set_loader_cclis_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools, training=False)
            # val_loader = set_valloader_co2l_tinyimagenet(opt=opt, normalize=normalize)
            # linear_loader = set_linearloader_co2l_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = None
            linear_loader = None
        
        # method_tools["subset_sample_num"] = subset_sample_num
        # method_tools["post_loader"] = post_loader
        model.module.subset_sample_num = subset_sample_num
        model.module.post_loader = post_loader
    
    elif opt.method in ["prco"]:

        if opt.dataset == "cifar10":
            train_loader, subset_indices, subset_sample_num = set_loader_prco_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices, model=model, training=True)
            val_loader = set_valloader_co2l_cifar10(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
        elif opt.dataset == "cifar100":
            train_loader, subset_indices, subset_sample_num = set_loader_prco_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices, model=model, training=True)
            val_loader = set_valloader_co2l_cifar100(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
        elif opt.dataset == "tiny-imagenet":
            train_loader, subset_indices, subset_sample_num = set_loader_prco_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices, model=model)
            # val_loader = set_valloader_co2l_tinyimagenet(opt=opt, normalize=normalize)
            # linear_loader = set_linearloader_co2l_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = None
            linear_loader = None


        


    

    # データ拡張も特に加えていない現在タスクのデータローダ
    # （gpmのメモリ更新などで普通の画像が必要な手法用）
    if opt.dataset == "cifar10":
        # vanilla_loader, _ = set_vanillaloader_er_cifar10(opt=opt, normalize=normalize)
        ncm_loader, _ = set_ncmloader_er_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
    elif opt.dataset == "cifar100":
        # vanilla_loader, _ = set_vanillaloader_er_cifar100(opt=opt, normalize=normalize)
        ncm_loader, _ = set_ncmloader_er_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
    elif opt.dataset == "tiny-imagenet":
        # vanilla_loader, _ = set_vanillaloader_er_tinyimagenet(opt=opt, normalize=normalize)
        # ncm_loader, _ = set_ncmloader_er_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
        ncm_loader = None
    

    # タスク増加シナリオにおける評価を行うためのデータローダ
    if opt.dataset == "cifar10":
        taskil_loaders = set_taskil_valloader_er_cifar10(opt=opt, normalize=normalize)
    elif opt.dataset == "cifar100":
        taskil_loaders = set_taskil_valloader_er_cifar100(opt=opt, normalize=normalize)
    elif opt.dataset == "tiny-imagenet":
        # taskil_loaders = set_taskil_valloader_er_tinyimagenet(opt=opt, normalize=normalize)
        taskil_loaders = None


    # タスク増加におけるknn分類を行うための訓練用データローダー
    if opt.dataset == "cifar10":
        knn_loaders = set_taskil_valloader_er_cifar10(opt=opt, normalize=normalize, train=True)
    elif opt.dataset == "cifar100":
        knn_loaders = set_taskil_valloader_er_cifar100(opt=opt, normalize=normalize, train=True)
    elif opt.dataset == "tiny-imagenet":
        # knn_loaders = set_taskil_valloader_er_tinyimagenet(opt=opt, normalize=normalize, train=True)
        knn_loaders = None



    dataloaders = {"train": train_loader, "val": val_loader, "linear": linear_loader, "ncm": ncm_loader, "taskil": taskil_loaders, "knn": knn_loaders}

    return dataloaders, subset_indices




# ==========================
# 検証用データローダーメインの作成
# ==========================
def set_loader_eval(opt, model, replay_indices, method_tools):

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

    if opt.dataset == "cifar10":
        val_loader = set_valloader_co2l_cifar10(opt=opt, normalize=normalize)
        linear_loader = set_linearloader_co2l_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
        ncm_loader, _ = set_ncmloader_er_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
        taskil_loaders = set_taskil_valloader_er_cifar10(opt=opt, normalize=normalize)
        knn_loaders = set_taskil_valloader_er_cifar10(opt=opt, normalize=normalize, train=True)
    elif opt.dataset == "cifar100":
        val_loader = set_valloader_co2l_cifar100(opt=opt, normalize=normalize)
        linear_loader = set_linearloader_co2l_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
        ncm_loader, _ = set_ncmloader_er_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
        taskil_loaders = set_taskil_valloader_er_cifar100(opt=opt, normalize=normalize)
        knn_loaders = set_taskil_valloader_er_cifar100(opt=opt, normalize=normalize, train=True)
    elif opt.dataset == 'tiny-imagenet':
        val_loader = set_valloader_co2l_tinyimagenet(opt=opt, normalize=normalize)
        linear_loader = set_linearloader_co2l_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
        ncm_loader, _ = set_ncmloader_er_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
        taskil_loaders = set_taskil_valloader_er_tinyimagenet(opt=opt, normalize=normalize)
        knn_loaders = set_taskil_valloader_er_tinyimagenet(opt=opt, normalize=normalize, train=True)
    


    dataloaders = {"val": val_loader, "linear": linear_loader, "ncm": ncm_loader, "taskil": taskil_loaders, "knn": knn_loaders}

    return dataloaders




# ==============================================================
# バッファの作成
# ==============================================================
def set_buffer(opt, model, prev_indices=None, method_tools=None):

    if opt.method in ["er"]:

        from dataloaders.buffer_er import set_replay_samples_reservoir
        from dataloaders.buffer_er import set_replay_samples_ring

        if opt.mem_type == "reservoir":
            replay_indices = set_replay_samples_reservoir(opt, model, prev_indices=prev_indices)
        elif opt.mem_type == "ring":
            replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices)
        else:
            assert False
    
    elif opt.method in ["co2l"]:

        from dataloaders.buffer_er import set_replay_samples_ring
        replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices) 

    elif opt.method in ["cclis"]:


        from dataloaders.buffer_cclis import set_replay_samples_cclis
        
        # importance_weight = method_tools["importance_weight"]
        importance_weight = model.module.importance_weight
        # score = method_tools["score"]
        score = model.module.score

        replay_indices, importance_weight, val_targets = set_replay_samples_cclis(
            opt, prev_indices=prev_indices, prev_importance_weight=importance_weight, prev_score=score
        )  # [prev_sample_num] tensor
        print("replauy_indices: ", replay_indices)
        # print("importance_weight: ", importance_weight)
        # print("val_targets: ", val_targets)
        # print("len(val_targets): ", len(val_targets))

        # method_tools["importance_weight"] = importance_weight
        # method_tools["val_targets"] = val_targets
        model.module.importance_weight = importance_weight
        model.module.val_targets = val_targets
    
    elif opt.method in ["prco"]:
        from dataloaders.buffer_er import set_replay_samples_ring
        replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices) 


    
    return replay_indices, method_tools