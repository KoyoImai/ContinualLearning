import os
import logging


from util import save_classifier, write_csv
from train.train_co2l import train_co2l, val_co2l, ncm_co2l
from train.train_cclis import train_cclis, val_cclis, ncm_cclis, adjust_learning_rate_cclis

logger = logging.getLogger(__name__)




def train(opt, model, model2, criterion, optimizer, scheduler, dataloader, epoch, method_tools):

    # データローダーの分解
    train_loader = dataloader["train"]
    val_loader = dataloader["val"]
    linear_loader = dataloader["linear"]
    ncm_loader = dataloader["ncm"]
    taskil_loaders = dataloader["taskil"]
    knn_train_loaders = dataloader["knn"]


    if opt.method == "er":

        assert False
    
    elif opt.method == "co2l":

        loss, model2 = train_co2l(opt=opt, model=model, model2=model2, criterion=criterion, optimizer=optimizer,
                                  scheduler=scheduler, train_loader=train_loader, epoch=epoch)
    
        if opt.eval and (epoch % opt.val_freq == 0):
            classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier = val_co2l(opt, model, model2, linear_loader, val_loader, taskil_loaders, knn_train_loaders, epoch)

            # 各タスクの精度を「task0 acc=100.00, task1 acc=90.00」の形式で整形
            taskil_acc_str = ', '.join([f"task{i} acc={acc:.2f}" for i, acc in enumerate(all_task_accuracies)])
            taskil_knnacc_str = ', '.join([f"task{i} knnacc={acc:.5f}" for i, acc in enumerate(all_task_knn_accuracies)])

            ncm_acc = ncm_co2l(model, ncm_loader, val_loader)

            logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                        ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}, \
                        {taskil_acc_str}, {taskil_knnacc_str}")
        
            # classifierの保存
            dir_path = f"{opt.model_path}/task{opt.target_task:02d}"
            file_path = f"{dir_path}/classifier_epoch{epoch:03d}.pth"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            save_classifier(classifier, opt, opt.epochs, file_path)

    elif opt.method == "cclis":

        adjust_learning_rate_cclis(opt, optimizer, epoch)

        subset_sample_num = model.module.subset_sample_num
        score_mask = model.module.score_mask

        loss, model2 = train_cclis(opt=opt, model=model, model2=model2, criterion=criterion, optimizer=optimizer,
                                   train_loader=train_loader, epoch=epoch, subset_sample_num=subset_sample_num, score_mask=score_mask)

        if opt.eval and (epoch % opt.val_freq == 0):
            classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier = val_cclis(opt, model, model2, linear_loader, val_loader, taskil_loaders, knn_train_loaders, epoch)

            # 各タスクの精度を「task0 acc=100.00, task1 acc=90.00」の形式で整形
            taskil_acc_str = ', '.join([f"task{i} acc={acc:.2f}" for i, acc in enumerate(all_task_accuracies)])
            taskil_knnacc_str = ', '.join([f"task{i} knnacc={acc:.5f}" for i, acc in enumerate(all_task_knn_accuracies)])

            ncm_acc = ncm_cclis(model, ncm_loader, val_loader)

            logger.info(f"task {opt.target_task} Epoch {epoch}: train_loss={loss:.4f}, \
                        ClassIL_accuracy={classil_acc:.3f}, TaskIL_accuracy={taskil_acc:.3f}, NCM_accuracy={ncm_acc:.3f}, \
                        {taskil_acc_str}, {taskil_knnacc_str}")
        
            # classifierの保存
            dir_path = f"{opt.model_path}/task{opt.target_task:02d}"
            file_path = f"{dir_path}/classifier_epoch{epoch:03d}.pth"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            save_classifier(classifier, opt, opt.epochs, file_path)

    
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

    elif opt.method == "cclis":

        classil_acc, taskil_acc, all_task_accuracies, all_task_knn_accuracies, all_task_losses, classifier = val_cclis(opt, model, None, linear_loader, val_loader, taskil_loaders, knn_train_loaders, opt.target_epoch)
        write_csv(classil_acc, opt.result_path, "classil_acc", opt.target_task, opt.target_epoch)
        write_csv(taskil_acc, opt.result_path, "taskil_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_accuracies, opt.result_path, "all_task_acc", opt.target_task, opt.target_epoch)
        write_csv(all_task_knn_accuracies, opt.result_path, "all_task_knn_acc", opt.target_task, opt.target_epoch)

