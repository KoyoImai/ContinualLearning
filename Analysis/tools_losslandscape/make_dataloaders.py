

from regex import F
from torchvision import transforms, datasets


from tools_losslandscape.cclis_dataloader import cclis_dataloader_cifar10
from tools_losslandscape.co2l_dataloader import co2l_dataloader_cifar10
from tools_losslandscape.er_dataloader import er_dataloader_cifar10


def make_dataloader(opt):

    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100': 
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

    
    
    if opt.method == "cclis" and (not opt.use_classifier):

        if opt.dataset == "cifar10":

            train_loader = cclis_dataloader_cifar10(opt=opt, normalize=normalize, training=True)
            val_loader = cclis_dataloader_cifar10(opt=opt, normalize=normalize, training=False)
        
        else:
            assert False
    
    elif opt.method == "co2l" and (not opt.use_classifier):

        if opt.dataset == "cifar10":

            train_loader = co2l_dataloader_cifar10(opt=opt, normalize=normalize, train=True)
            val_loader = co2l_dataloader_cifar10(opt=opt, normalize=normalize, train=False)
        
        else:
            assert False

    elif opt.method == "er" or opt.use_classifier:

        if opt.dataset == "cifar10":
            train_loader = er_dataloader_cifar10(opt=opt, normalize=normalize, train=True)
            val_loader = er_dataloader_cifar10(opt=opt, normalize=normalize, train=False)


    return train_loader, val_loader