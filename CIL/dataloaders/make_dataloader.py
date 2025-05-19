
from regex import F
from torchvision import transforms, datasets

# er
from dataloaders.dataloader_er import set_loader_er_cifar10, set_valloader_er_cifar10
from dataloaders.dataloader_er import set_loader_er_cifar100, set_valloader_er_cifar100
from dataloaders.dataloader_er import set_loader_er_tinyimagenet, set_valloader_er_tinyimagenet
from dataloaders.dataloader_er import set_vanillaloader_er_cifar10, set_vanillaloader_er_cifar100, set_vanillaloader_er_tinyimagenet
from dataloaders.dataloader_er import set_ncmloader_er_cifar10, set_ncmloader_er_cifar100, set_ncmloader_er_tinyimagenet
from dataloaders.dataloader_er import set_taskil_valloader_er_cifar10, set_taskil_valloader_er_cifar100, set_taskil_valloader_er_tinyimagenet

# co2l
from dataloaders.dataloader_co2l import set_loader_co2l_cifar10, set_linearloader_co2l_cifar10, set_valloader_co2l_cifar10
from dataloaders.dataloader_co2l import set_loader_co2l_cifar100, set_linearloader_co2l_cifar100, set_valloader_co2l_cifar100
from dataloaders.dataloader_co2l import set_loader_co2l_tinyimagenet, set_linearloader_co2l_tinyimagenet, set_valloader_co2l_tinyimagenet

# gpm
from dataloaders.dataloader_gpm import set_loader_gpm_cifar10, set_valloader_gpm_cifar10
from dataloaders.dataloader_gpm import set_loader_gpm_cifar100, set_valloader_gpm_cifar100
from dataloaders.dataloader_gpm import set_loader_gpm_tinyimagenet, set_valloader_gpm_tinyimagenet


# lucir
from dataloaders.dataloader_lucir import set_loader_lucir_cifar10, set_valloader_lucir_cifar10
from dataloaders.dataloader_lucir import set_loader_lucir_cifar100, set_valloader_lucir_cifar100
from dataloaders.dataloader_lucir import set_loader_lucir_tinyimagenet, set_valloader_lucir_tinyimagenet

# fs-dgpm
from dataloaders.dataloader_fsdgpm import set_loader_fsdgpm_cifar10, set_valloader_fsdgpm_cifar10
from dataloaders.dataloader_fsdgpm import set_loader_fsdgpm_cifar100, set_valloader_fsdgpm_cifar100
from dataloaders.dataloader_fsdgpm import set_loader_fsdgpm_tinyimagenet, set_valloader_fsdgpm_tinyimagenet

# cclis
from dataloaders.dataloader_cclis import set_loader_cclis_cifar10, set_loader_cclis_cifar100, set_loader_cclis_tinyimagenet

# cclis-wo-ss
from dataloaders.dataloader_cclis_wo_ss import set_loader_cclis_wo_ss_cifar10, set_loader_cclis_wo_ss_cifar100, set_loader_cclis_wo_ss_tinyimagenet

# joint
from dataloaders.dataloader_joint import set_loader_joint_cifar10, set_valloader_joint_cifar10, set_linearloader_joint_cifar10
from dataloaders.dataloader_joint import set_loader_joint_cifar100, set_valloader_joint_cifar100, set_linearloader_joint_cifar100
from dataloaders.dataloader_joint import set_loader_joint_tinyimagenet, set_valloader_joint_tinyimagenet, set_linearloader_joint_tinyimagenet

# cclis-bw
from dataloaders.dataloader_cclis_bw import set_loader_cclis_bw_cifar10, set_loader_cclis_bw_cifar100, set_loader_cclis_bw_tinyimagenet





def set_loader(opt, replay_indices, method_tools):

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


    # 手法毎にデータローダーの作成方法を変更する（使用するデータ拡張，データの取り出し方が異なるので）
    if opt.method == "er":
        if opt.dataset == "cifar10":
            train_loader, subset_indices = set_loader_er_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_er_cifar10(opt=opt, normalize=normalize)
            linear_loader = None
        elif opt.dataset == "cifar100":
            train_loader, subset_indices = set_loader_er_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_er_cifar100(opt=opt, normalize=normalize)
            linear_loader = None
        elif opt.dataset == "tiny-imagenet":
            train_loader, subset_indices = set_loader_er_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_er_tinyimagenet(opt=opt, normalize=normalize)
            linear_loader = None
        else:
            assert False
    
    elif opt.method in ["co2l", "supcon", "simclr"]:
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
            val_loader = set_valloader_co2l_tinyimagenet(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)

    elif opt.method == "gpm":
        if opt.dataset == "cifar10":
            # train_loader, subset_indices = set_loader_er_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
            train_loader, subset_indices = set_loader_gpm_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_gpm_cifar10(opt=opt, normalize=normalize)
            linear_loader = None
        elif opt.dataset == "cifar100":
            # train_loader, subset_indices = set_loader_er_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
            train_loader, subset_indices = set_loader_gpm_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_gpm_cifar100(opt=opt, normalize=normalize)
            linear_loader = None
        elif opt.dataset == "tiny-imagenet":
            # train_loader, subset_indices = set_loader_er_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            train_loader, subset_indices = set_loader_gpm_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_gpm_tinyimagenet(opt=opt, normalize=normalize)
            linear_loader = None
        else:
            assert False
    
    elif opt.method == "lucir":
        if opt.dataset == "cifar10":
            train_loader, subset_indices = set_loader_lucir_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_lucir_cifar10(opt=opt, normalize=normalize)
            linear_loader = None
        elif opt.dataset == "cifar100":
            train_loader, subset_indices = set_loader_lucir_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_lucir_cifar100(opt=opt, normalize=normalize)
            linear_loader = None
        elif opt.dataset == "tiny-imagenet":
            train_loader, subset_indices = set_loader_lucir_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_lucir_tinyimagenet(opt=opt, normalize=normalize)
            linear_loader = None
        else:
            assert False

    elif opt.method == "fs-dgpm":
        if opt.dataset == "cifar10":
            # train_loader, subset_indices = set_loader_er_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
            train_loader, subset_indices = set_loader_fsdgpm_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_fsdgpm_cifar10(opt=opt, normalize=normalize)
            linear_loader = None
        elif opt.dataset == "cifar100":
            # train_loader, subset_indices = set_loader_er_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
            train_loader, subset_indices = set_loader_fsdgpm_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_fsdgpm_cifar100(opt=opt, normalize=normalize)
            linear_loader = None
        elif opt.dataset == "tiny-imagenet":
            # train_loader, subset_indices = set_loader_er_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            train_loader, subset_indices = set_loader_fsdgpm_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
            val_loader = set_valloader_fsdgpm_tinyimagenet(opt=opt, normalize=normalize)
            linear_loader = None
        else:
            assert False
    
    elif opt.method in ["cclis", "cclis-wo", "cclis-wo-is"]:

        if opt.dataset == "cifar10":
            train_loader, subset_indices, subset_sample_num = set_loader_cclis_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools)
            post_loader, _, _ = set_loader_cclis_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools, training=False)
            val_loader = set_valloader_co2l_cifar10(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
        elif opt.dataset == "cifar100":
            train_loader, subset_indices, subset_sample_num = set_loader_cclis_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools)
            post_loader, _, _ = set_loader_cclis_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools, training=False)
            val_loader = set_valloader_co2l_cifar100(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
        elif opt.dataset == 'tiny-imagenet':
            train_loader, subset_indices, subset_sample_num = set_loader_cclis_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools)
            post_loader, _, _ = set_loader_cclis_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools, training=False)
            val_loader = set_valloader_co2l_tinyimagenet(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
        
        method_tools["subset_sample_num"] = subset_sample_num
        method_tools["post_loader"] = post_loader

    elif opt.method in ["supcon-joint"]:
        if opt.dataset == "cifar10":
            train_loader, subset_indices = set_loader_joint_cifar10(opt=opt, normalize=normalize)
            val_loader = set_valloader_joint_cifar10(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_joint_cifar10(opt=opt, normalize=normalize)
        elif opt.dataset == "cifar100":
            train_loader, subset_indices = set_loader_joint_cifar100(opt=opt, normalize=normalize)
            val_loader = set_valloader_joint_cifar100(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_joint_cifar100(opt=opt, normalize=normalize)
        elif opt.dataset == 'tiny-imagenet':
            train_loader, subset_indices = set_loader_joint_tinyimagenet(opt=opt, normalize=normalize)
            val_loader = set_valloader_joint_tinyimagenet(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_joint_tinyimagenet(opt=opt, normalize=normalize)

    elif opt.method in ["cclis-wo-ss"]:

        if opt.dataset == "cifar10":
            train_loader, subset_indices, subset_sample_num = set_loader_cclis_wo_ss_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools)
            post_loader, _, _ = set_loader_cclis_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools, training=False)
            val_loader = set_valloader_co2l_cifar10(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
        elif opt.dataset == "cifar100":
            train_loader, subset_indices, subset_sample_num = set_loader_cclis_wo_ss_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools)
            post_loader, _, _ = set_loader_cclis_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools, training=False)
            val_loader = set_valloader_co2l_cifar100(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
        elif opt.dataset == 'tiny-imagenet':
            train_loader, subset_indices, subset_sample_num = set_loader_cclis_wo_ss_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools)
            post_loader, _, _ = set_loader_cclis_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools, training=False)
            val_loader = set_valloader_co2l_tinyimagenet(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
        
        method_tools["subset_sample_num"] = subset_sample_num
        method_tools["post_loader"] = post_loader
    
    elif opt.method in ["cclis-bw"]:

        if opt.dataset == "cifar10":
            train_loader, subset_indices, subset_sample_num = set_loader_cclis_bw_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools)
            post_loader, _, _ = set_loader_cclis_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools, training=False)
            val_loader = set_valloader_co2l_cifar10(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
        elif opt.dataset == "cifar100":
            train_loader, subset_indices, subset_sample_num = set_loader_cclis_bw_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools)
            post_loader, _, _ = set_loader_cclis_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools, training=False)
            val_loader = set_valloader_co2l_cifar100(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
        elif opt.dataset == 'tiny-imagenet':
            train_loader, subset_indices, subset_sample_num = set_loader_cclis_bw_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools)
            post_loader, _, _ = set_loader_cclis_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices, method_tools=method_tools, training=False)
            val_loader = set_valloader_co2l_tinyimagenet(opt=opt, normalize=normalize)
            linear_loader = set_linearloader_co2l_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)
        
        method_tools["subset_sample_num"] = subset_sample_num
        method_tools["post_loader"] = post_loader

    else:
        assert False


    # データ拡張も特に加えていない現在タスクのデータローダ
    # （gpmのメモリ更新などで普通の画像が必要な手法用）
    if opt.dataset == "cifar10":
        vanilla_loader, _ = set_vanillaloader_er_cifar10(opt=opt, normalize=normalize)
        ncm_loader, _ = set_ncmloader_er_cifar10(opt=opt, normalize=normalize, replay_indices=replay_indices)
    elif opt.dataset == "cifar100":
        vanilla_loader, _ = set_vanillaloader_er_cifar100(opt=opt, normalize=normalize)
        ncm_loader, _ = set_ncmloader_er_cifar100(opt=opt, normalize=normalize, replay_indices=replay_indices)
    elif opt.dataset == "tiny-imagenet":
        vanilla_loader, _ = set_vanillaloader_er_tinyimagenet(opt=opt, normalize=normalize)
        ncm_loader, _ = set_ncmloader_er_tinyimagenet(opt=opt, normalize=normalize, replay_indices=replay_indices)


    # タスク増加シナリオにおける評価を行うためのデータローダ
    if opt.dataset == "cifar10":
        taskil_loaders = set_taskil_valloader_er_cifar10(opt=opt, normalize=normalize)
    elif opt.dataset == "cifar100":
        taskil_loaders = set_taskil_valloader_er_cifar100(opt=opt, normalize=normalize)
    elif opt.dataset == "tiny-imagenet":
        taskil_loaders = set_taskil_valloader_er_tinyimagenet(opt=opt, normalize=normalize)

    dataloader = {"train": train_loader, "linear": linear_loader, "val": val_loader, "vanilla": vanilla_loader, "ncm": ncm_loader, "taskil": taskil_loaders}
    
    return dataloader, subset_indices, method_tools