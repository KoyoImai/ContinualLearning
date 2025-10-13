import os
import logging


from util import save_classifier, write_csv
from train.train_co2l import train_co2l, val_co2l, ncm_co2l, val_co2l4timnet
from train.train_cclis import train_cclis, val_cclis, ncm_cclis, adjust_learning_rate_cclis, val_cclis4timnet
from train.train_prco import train_prco, adjust_learning_rate_prco
from train.train_prco_fimcl import train_prco_fimcl
from train.train_prco_fimclv2 import train_prco_fimclv2
from train.train_prco_progefm import train_prco_progefm, update_efn


logger = logging.getLogger(__name__)




def train(opt, model, model2, criterion, optimizer, scheduler, dataloader, epoch, method_tools):

    # データローダーの分解
    train_loader = dataloader["train"]
    val_loader = dataloader["val"]
    linear_loader = dataloader["linear"]
    ncm_loader = dataloader["ncm"]
    taskil_loaders = dataloader["taskil"]
    knn_train_loaders = dataloader["knn"]
    replay_loader = dataloader["replay"]


    if opt.method == "er":

        assert False
    
    elif opt.method == "co2l":

        loss, model2 = train_co2l(opt=opt, model=model, model2=model2, criterion=criterion, optimizer=optimizer,
                                  scheduler=scheduler, train_loader=train_loader, epoch=epoch)
    

    elif opt.method == "cclis":

        adjust_learning_rate_cclis(opt, optimizer, epoch)

        subset_sample_num = model.module.subset_sample_num
        score_mask = model.module.score_mask

        loss, model2 = train_cclis(opt=opt, model=model, model2=model2, criterion=criterion, optimizer=optimizer,
                                   train_loader=train_loader, epoch=epoch, subset_sample_num=subset_sample_num, score_mask=score_mask)
        

    elif opt.method in ["prco", "prco-efm"]:

        adjust_learning_rate_cclis(opt, optimizer, epoch)

        loss, model2 = train_prco(opt=opt, model=model, model2=model2, criterion=criterion,
                                  optimizer=optimizer, train_loader=train_loader, epoch=epoch)
    

    elif opt.method == "prco-fimcl":

        adjust_learning_rate_cclis(opt, optimizer, epoch)

        loss, model2 = train_prco_fimcl(opt=opt, model=model, model2=model2, criterion=criterion,
                                        optimizer=optimizer, train_loader=train_loader, epoch=epoch)


    elif opt.method == "prco-fimclv2":

        adjust_learning_rate_cclis(opt, optimizer, epoch)

        loss, model2 = train_prco_fimclv2(opt=opt, model=model, model2=model2, criterion=criterion,
                                          optimizer=optimizer, train_loader=train_loader, epoch=epoch)
    

    elif opt.method == "prco-fimclv3":

        adjust_learning_rate_cclis(opt, optimizer, epoch)

        if (epoch > 100) and opt.target_task != 0:
            cal_fim = True
        else:
            cal_fim = False

        loss, model2 = train_prco_fimclv2(opt=opt, model=model, model2=model2, criterion=criterion,
                                          optimizer=optimizer, train_loader=train_loader, epoch=epoch, cal_fim=False)

    
    elif opt.method in ["prco-progefm"]:

        adjust_learning_rate_cclis(opt, optimizer, epoch)

        loss, model2 = train_prco_progefm(opt=opt, model=model, model2=model2, criterion=criterion,
                                          optimizer=optimizer, train_loader=train_loader, epoch=epoch)
        
        if (epoch % opt.update_efm_freq == 0) and (opt.target_task != 0):
            update_efn(opt=opt, model=model, train_loader=replay_loader, feat=True)

        
    else:

        assert False



def eval(model, dataloader, opt):

    # データローダーの分解
    val_loader = dataloader["val"]
    linear_loader = dataloader["linear"]
    ncm_loader = dataloader["ncm"]
    taskil_loaders = dataloader["taskil"]
    knn_train_loaders = dataloader["knn"]


    if opt.method == "er":

        assert False
    
    elif opt.method == "co2l":

        classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier = val_co2l(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        write_csv(classil_acc, opt.result_path, "classil_acc", opt.target_task, opt.target_epoch)
        write_csv(taskil_acc, opt.result_path, "taskil_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_accuracies, opt.result_path, "all_task_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_knn_accuracies, opt.result_path, "all_task_knn_acc", opt.target_task, opt.target_epoch)

    elif opt.method == "cclis":

        classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier = val_cclis(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        write_csv(classil_acc, opt.result_path, "classil_acc", opt.target_task, opt.target_epoch)
        write_csv(taskil_acc, opt.result_path, "taskil_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_accuracies, opt.result_path, "all_task_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_knn_accuracies, opt.result_path, "all_task_knn_acc", opt.target_task, opt.target_epoch)
    

    elif opt.method in ["prco", "prco-efm"]:

        classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier = val_cclis(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        write_csv(classil_acc, opt.result_path, "classil_acc", opt.target_task, opt.target_epoch)
        write_csv(taskil_acc, opt.result_path, "taskil_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_accuracies, opt.result_path, "all_task_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_knn_accuracies, opt.result_path, "all_task_knn_acc", opt.target_task, opt.target_epoch)
    

    elif opt.method in ["prco-fimcl", "prco-fimclv2", 'prco-fimclv3']:

        classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier = val_cclis(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        write_csv(classil_acc, opt.result_path, "classil_acc", opt.target_task, opt.target_epoch)
        write_csv(taskil_acc, opt.result_path, "taskil_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_accuracies, opt.result_path, "all_task_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_knn_accuracies, opt.result_path, "all_task_knn_acc", opt.target_task, opt.target_epoch)




def eval_ncm(model, dataloader, opt):

    # データローダーの分解
    val_loader = dataloader["val"]
    linear_loader = dataloader["linear"]
    ncm_loader = dataloader["ncm"]
    taskil_loaders = dataloader["taskil"]

    if opt.method == "er":

        assert False
    
    elif opt.method == "co2l":

        acc_euclidean, acc_cosine = ncm_co2l(opt=opt, model=model, ncm_loader=ncm_loader, val_loader=val_loader)
        write_csv(acc_euclidean, opt.result_path, "ncm_euclidean_acc", opt.target_task, opt.target_epoch)
        write_csv(acc_cosine, opt.result_path, "ncm_cosine_acc", opt.target_task, opt.target_epoch)
    
    elif opt.method == "cclis":
        
        acc_euclidean, acc_cosine, task_acc_euclidean, task_acc_cosine = ncm_cclis(opt=opt, model=model, ncm_loader=ncm_loader, val_loader=val_loader)
        # taskil_ncmacc_euclidwan = ', '.join([f"task{i} acc={acc:.2f}" for i, acc in enumerate(task_acc_euclidean)])
        # taskil_ncmacc_cosine = ', '.join([f"task{i} acc={acc:.2f}" for i, acc in enumerate(task_acc_cosine)])
        write_csv(acc_euclidean, opt.result_path, "ncm_euclidean_acc", opt.target_task, opt.target_epoch)
        write_csv(acc_cosine, opt.result_path, "ncm_cosine_acc", opt.target_task, opt.target_epoch)
        write_csv(task_acc_euclidean, opt.result_path, "ncm_taskil_euclidean_acc", opt.target_task, opt.target_epoch)
        write_csv(task_acc_cosine, opt.result_path, "ncm_taskil_cosine_acc", opt.target_task, opt.target_epoch)

    elif opt.method in ["prco", 'prco-fimclv2', 'prco-efm']:

        acc_euclidean, acc_cosine, task_acc_euclidean, task_acc_cosine = ncm_cclis(opt=opt, model=model, ncm_loader=ncm_loader, val_loader=val_loader)
        write_csv(acc_euclidean, opt.result_path, "ncm_euclidean_acc", opt.target_task, opt.target_epoch)
        write_csv(acc_cosine, opt.result_path, "ncm_cosine_acc", opt.target_task, opt.target_epoch)
        write_csv(task_acc_euclidean, opt.result_path, "ncm_taskil_euclidean_acc", opt.target_task, opt.target_epoch)
        write_csv(task_acc_cosine, opt.result_path, "ncm_taskil_cosine_acc", opt.target_task, opt.target_epoch)




    else:

        assert False



def eval4timnet(model, dataloader, opt):

    # データローダーの分解
    val_loader = dataloader["val"]
    linear_loader = dataloader["linear"]
    # ncm_loader = dataloader["ncm"]
    taskil_loaders = dataloader["taskil"]
    knn_train_loaders = dataloader["knn"]


    if opt.method == "er":

        assert False
    
    elif opt.method == "co2l":

        # classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier = val_co2l(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, classifier = val_co2l4timnet(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        write_csv(classil_acc, opt.result_path, "classil_acc", opt.target_task, opt.target_epoch)
        write_csv(taskil_acc, opt.result_path, "taskil_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_accuracies, opt.result_path, "all_task_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_knn_accuracies, opt.result_path, "all_task_knn_acc", opt.target_task, opt.target_epoch)

    elif opt.method in ["cclis"]:

        # classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier = val_cclis(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, classifier = val_cclis4timnet(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        write_csv(classil_acc, opt.result_path, "classil_acc", opt.target_task, opt.target_epoch)
        write_csv(taskil_acc, opt.result_path, "taskil_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_accuracies, opt.result_path, "all_task_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_knn_accuracies, opt.result_path, "all_task_knn_acc", opt.target_task, opt.target_epoch)

    elif opt.method in ["prco", 'prco-efm']:

        # classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier = val_cclis(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, classifier = val_cclis4timnet(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        write_csv(classil_acc, opt.result_path, "classil_acc", opt.target_task, opt.target_epoch)
        write_csv(taskil_acc, opt.result_path, "taskil_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_accuracies, opt.result_path, "all_task_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_knn_accuracies, opt.result_path, "all_task_knn_acc", opt.target_task, opt.target_epoch)
    

    elif opt.method in ["prco-fimcl", "prco-fimclv2", 'prco-fimclv3']:

        # classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier = val_cclis(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, classifier = val_cclis4timnet(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        write_csv(classil_acc, opt.result_path, "classil_acc", opt.target_task, opt.target_epoch)
        write_csv(taskil_acc, opt.result_path, "taskil_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_accuracies, opt.result_path, "all_task_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_knn_accuracies, opt.result_path, "all_task_knn_acc", opt.target_task, opt.target_epoch)

        