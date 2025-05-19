import logging
import copy

import torch.optim.lr_scheduler as lr_scheduler

# from survey.CIL.dataloaders.dataloader_er import set_taskil_valloader_er_cifar10
from trains.train_er import train_er, val_er, ncm_er, taskil_val_er
from trains.train_co2l import ncm_co2l, train_co2l, val_co2l
from trains.train_gpm import train_gpm_first, train_gpm_other, val_gpm, ncm_gpm
from trains.train_lucir import train_lucir, val_lucir, ncm_lucir
from trains.train_fsdgpm import train_fsdgpm, val_fsdgpm
from trains.train_cclis import train_cclis, val_cclis, ncm_cclis, adjust_learning_rate_cclis
from trains.train_cclis_wo import train_cclis_wo, val_cclis_wo, ncm_cclis_wo
from trains.train_supcon import train_supcon
from trains.train_cclis_bw import train_cclis_bw



logger = logging.getLogger(__name__)


def train(opt, model, model2, criterion, optimizer, scheduler, dataloader, epoch, method_tools):

    # データローダーの分解
    train_loader = dataloader["train"]
    val_loader = dataloader["val"]
    linear_loader = dataloader["linear"]
    vanilla_loader = dataloader["vanilla"]
    ncm_loader = dataloader["ncm"]
    taskil_loaders = dataloader["taskil"]

    
    if opt.method == "er":
        loss, _ = train_er(opt, model, model2, criterion, optimizer,
                           scheduler, train_loader, val_loader, epoch)
        classil_acc, taskil_acc = val_er(opt, model, model2, criterion,
                                         optimizer, scheduler, train_loader, val_loader, epoch)
        ncm_acc = ncm_er(model, ncm_loader, val_loader)

        all_task_accuracies, all_task_losses = taskil_val_er(opt, model, criterion, taskil_loaders)
        # 各タスクの精度を「task0 acc=100.00, task1 acc=90.00」の形式で整形
        taskil_acc_str = ', '.join([f"task{i} acc={acc:.2f}" for i, acc in enumerate(all_task_accuracies)])

        print("all_task_accuracies: ", all_task_accuracies)
        logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                    ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}, \
                    {taskil_acc_str}")



    elif opt.method == "co2l":
        
        loss, model2 = train_co2l(opt=opt, model=model, model2=model2,
                                  criterion=criterion, optimizer=optimizer,
                                  scheduler=scheduler, train_loader=train_loader, epoch=epoch)
        if epoch % 50 == 0:
            classil_acc, taskil_acc, all_task_accuracies, all_task_losses = val_co2l(opt, model, model2, linear_loader, val_loader, taskil_loaders, epoch)
            # 各タスクの精度を「task0 acc=100.00, task1 acc=90.00」の形式で整形
            taskil_acc_str = ', '.join([f"task{i} acc={acc:.2f}" for i, acc in enumerate(all_task_accuracies)])

            ncm_acc = ncm_co2l(model, ncm_loader, val_loader)

            logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                        ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}, \
                        {taskil_acc_str}")
    
    elif opt.method == "gpm":

        if opt.target_task == 0:
            loss = train_gpm_first(opt, model, criterion, optimizer, scheduler, train_loader, epoch)
        else:
            loss = train_gpm_other(opt=opt, model=model, criterion=criterion, optimizer=optimizer,
                                   scheduler=scheduler, train_loader=train_loader, epoch=epoch, method_tools=method_tools)
        
        classil_acc, taskil_acc = val_gpm(opt, model, criterion, optimizer, scheduler, train_loader, val_loader, epoch)
        ncm_acc = ncm_gpm(model, ncm_loader, val_loader)

        all_task_accuracies, all_task_losses = taskil_val_er(opt, model, criterion, taskil_loaders)
        # 各タスクの精度を「task0 acc=100.00, task1 acc=90.00」の形式で整形
        taskil_acc_str = ', '.join([f"task{i} acc={acc:.2f}" for i, acc in enumerate(all_task_accuracies)])

        logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                    ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}, \
                    {taskil_acc_str}")

    elif opt.method == "lucir":

        scheduler.step()
        cur_lamda = method_tools['cur_lamda']
        loss, model2 = train_lucir(opt, model, model2, criterion, optimizer,
                                   scheduler, train_loader, val_loader, epoch, opt.temp, cur_lamda)
        classil_acc, taskil_acc = val_lucir(opt, model, model2, criterion, optimizer,
                                            scheduler, train_loader, val_loader, epoch)
        ncm_acc = ncm_lucir(model, ncm_loader, val_loader)

        logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                    ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}")
    
    elif opt.method == "fs-dgpm":
        
        loss = train_fsdgpm(opt, model, model2, criterion, optimizer, scheduler, train_loader, epoch)

        if opt.earlystop:
            assert False
        
        classil_acc, taskil_acc = val_fsdgpm(opt, model, model2, criterion, optimizer, scheduler, train_loader, val_loader, epoch)
        # logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
        #             ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}")
        
        all_task_accuracies, all_task_losses = taskil_val_er(opt, model, criterion, taskil_loaders)
        # 各タスクの精度を「task0 acc=100.00, task1 acc=90.00」の形式で整形
        taskil_acc_str = ', '.join([f"task{i} acc={acc:.2f}" for i, acc in enumerate(all_task_accuracies)])
        
        logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                    ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, \
                    {taskil_acc_str}")

    elif opt.method in ["cclis"]:

        adjust_learning_rate_cclis(opt, optimizer, epoch)

        subset_sample_num = method_tools["subset_sample_num"]
        score_mask = method_tools["score_mask"]

        loss, model2 = train_cclis(opt=opt, model=model, model2=model2,
                                   criterion=criterion, optimizer=optimizer,
                                   subset_sample_num=subset_sample_num, score_mask=score_mask,
                                   scheduler=scheduler, train_loader=train_loader, epoch=epoch)
        if epoch % 100 == 0:
            classil_acc, taskil_acc, all_task_accuracies, all_task_losses = val_cclis(opt, model, model2, linear_loader, val_loader, taskil_loaders, epoch)
            # 各タスクの精度を「task0 acc=100.00, task1 acc=90.00」の形式で整形
            taskil_acc_str = ', '.join([f"task{i} acc={acc:.2f}" for i, acc in enumerate(all_task_accuracies)])

            ncm_acc = ncm_cclis(model, ncm_loader, val_loader)

            logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                        ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}, \
                        {taskil_acc_str}")
    
    elif opt.method in ["supcon-joint", "supcon"]:

        loss, model2 = train_supcon(opt=opt, model=model, model2=model2,
                                    criterion=criterion, optimizer=optimizer,
                                    scheduler=scheduler, train_loader=train_loader, epoch=epoch)
        
        if epoch % 50 == 0 or epoch == opt.epochs or epoch == opt.start_epoch:
            classil_acc, taskil_acc, all_task_accuracies, all_task_losses = val_co2l(opt, model, model2, linear_loader, val_loader, taskil_loaders, epoch)

            taskil_acc_str = ', '.join([f"task{i} acc={acc:.2f}" for i, acc in enumerate(all_task_accuracies)])

            ncm_acc = ncm_co2l(model, ncm_loader, val_loader)

            logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                        ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}, \
                        {taskil_acc_str}")
    
    elif opt.method in ["cclis-wo", "cclis-wo-ss", "cclis-wo-is"]:

        subset_sample_num = method_tools["subset_sample_num"]
        score_mask = method_tools["score_mask"]

        loss, model2 = train_cclis_wo(opt=opt, model=model, model2=model2,
                                      criterion=criterion, optimizer=optimizer,
                                      subset_sample_num=subset_sample_num, score_mask=score_mask,
                                      scheduler=scheduler, train_loader=train_loader, epoch=epoch)
        if epoch % 50 == 0:
            classil_acc, taskil_acc, all_task_accuracies, all_task_losses = val_cclis_wo(opt, model, model2, linear_loader, val_loader, taskil_loaders, epoch)
            # 各タスクの精度を「task0 acc=100.00, task1 acc=90.00」の形式で整形
            taskil_acc_str = ', '.join([f"task{i} acc={acc:.2f}" for i, acc in enumerate(all_task_accuracies)])

            ncm_acc = ncm_cclis_wo(model, ncm_loader, val_loader)

            logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                        ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}, \
                        {taskil_acc_str}")
    
    elif opt.method in ["cclis-bw"]:

        adjust_learning_rate_cclis(opt, optimizer, epoch)

        subset_sample_num = method_tools["subset_sample_num"]
        score_mask = method_tools["score_mask"]
        criterion_bw = method_tools["criterion_bw"]

        loss, model2 = train_cclis_bw(opt=opt, model=model, model2=model2,
                                      criterion=criterion, criterion_bw=criterion_bw, optimizer=optimizer,
                                      subset_sample_num=subset_sample_num, score_mask=score_mask,
                                      scheduler=scheduler, train_loader=train_loader, epoch=epoch)
        
        if epoch % 100 == 0:
            classil_acc, taskil_acc, all_task_accuracies, all_task_losses = val_cclis(opt, model, model2, linear_loader, val_loader, taskil_loaders, epoch)
            # 各タスクの精度を「task0 acc=100.00, task1 acc=90.00」の形式で整形
            taskil_acc_str = ', '.join([f"task{i} acc={acc:.2f}" for i, acc in enumerate(all_task_accuracies)])

            ncm_acc = ncm_cclis(model, ncm_loader, val_loader)

            logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                        ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}, \
                        {taskil_acc_str}")

    else:
        assert False
        

    