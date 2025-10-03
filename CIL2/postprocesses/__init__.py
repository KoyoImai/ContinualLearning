
import os

from util import save_model

from postprocesses.postprocess_cclis import postprocess_cclis
from postprocesses.postprocess_prco import postprocess_prco
from postprocesses.postprocess_prco_fimcl import postprocess_prco as postprocess_prco_fimcl



def post_process(opt, model, model2, dataloader, criterion, optimizer, method_tools, replay_indices):

    # # データローダーの分解
    train_loader = dataloader["train"]
    linear_loader = dataloader["linear"]
    # train_loader = model.debug_loader
    # val_loader = dataloader["val"]
    # vanilla_loader = dataloader["vanilla"]


    if opt.method in ["er", "co2l", "lucir", "supcon", "supcon-joint", "simclr"]:
        return
    
    elif opt.method == "cclis":
        postprocess_cclis(opt, model, model2, criterion, replay_indices)
        return 

    elif opt.method in ["prco"]:
        if opt.distill_type == "EFC":
            postprocess_prco(opt=opt, model=model, train_loader=linear_loader)
        return 

    elif opt.method in ["prco-fimcl", "prco-fimclv3"]:

        if opt.distill_type == "EFC":
            postprocess_prco_fimcl(opt=opt, model=model, train_loader=linear_loader)
            model.module.update_efm(opt=opt)
        
        else:

            # prco-fimclv2 ではEFC正則化以外は使用しないので，ここには来ないはず．
            assert False
        return 
    
    elif opt.method in ["prco-fimclv2"]:

        if opt.target_task != 0:
            
            # 追加学習を実行
            from train.train_prco_fimclv2 import train_prco_fimclv2, adjust_learning_rate_prco_addlearning
            for epoch in range(1, opt.add_epoch+1):

                adjust_learning_rate_prco_addlearning(opt, optimizer, epoch)

                loss, model2 = train_prco_fimclv2(opt=opt, model=model, model2=model2, criterion=criterion,
                                                    optimizer=optimizer, train_loader=train_loader, epoch=epoch, cal_fim=True)
                
                if opt.epoch_save:
                    dir_path = f"{opt.model_path}/task{opt.target_task:02d}"
                    file_path = f"{dir_path}/model_epoch{opt.epochs + epoch:03d}.pth"
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    save_model(model, method_tools["optimizer"], opt, opt.epochs, file_path)



        if opt.distill_type == "EFC":
            postprocess_prco_fimcl(opt=opt, model=model, train_loader=linear_loader)
            model.module.update_efm(opt=opt)
        
        else:

            # prco-fimclv2 ではEFC正則化以外は使用しないので，ここには来ないはず．
            assert False
        return 


    
    else:
        assert False

