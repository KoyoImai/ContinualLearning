


from postprocesses.postprocess_cclis import postprocess_cclis

def post_process(opt, model, model2, dataloader, criterion, method_tools, replay_indices):

    # # データローダーの分解
    # train_loader = dataloader["train"]
    # val_loader = dataloader["val"]
    # linear_loader = dataloader["linear"]
    # vanilla_loader = dataloader["vanilla"]


    if opt.method in ["er", "co2l", "lucir", "supcon", "supcon-joint", "simclr"]:
        return
    
    elif opt.method == "cclis":
        postprocess_cclis(opt, model, model2, criterion, replay_indices)
        return 

    else:
        assert False

