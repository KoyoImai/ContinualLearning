


from postprocesses.postprocess_cclis import postprocess_cclis
from postprocesses.postprocess_prco import postprocess_prco



def post_process(opt, model, model2, dataloader, criterion, method_tools, replay_indices):

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
    
    else:
        assert False

