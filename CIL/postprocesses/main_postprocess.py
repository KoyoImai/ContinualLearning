
import numpy as np

from postprocesses.postprocess_gpm import postprocess_gpm
from postprocesses.postprocess_fsdgpm import postprocess_fsdgpm
from postprocesses.postprocess_cclis import postprocess_cclis



def post_process(opt, model, model2, dataloader, criterion, method_tools, replay_indices):

    # データローダーの分解
    train_loader = dataloader["train"]
    val_loader = dataloader["val"]
    linear_loader = dataloader["linear"]
    vanilla_loader = dataloader["vanilla"]


    if opt.method in ["er", "co2l", "lucir", "supcon", "supcon-joint", "simclr"]:
        return method_tools, model2
    elif opt.method == "gpm":

        # threshold = method_tools["threshold"]
        feature_list = method_tools["feature_list"]
        threshold = np.array([opt.threshold] * 20)

        # print("threshold: ", threshold)
        # print("feature_list: ", feature_list)

        # メモリの更新（feature_listの更新）
        feature_list = postprocess_gpm(opt=opt, model=model, vanilla_loader=vanilla_loader,
                                       feature_list=feature_list, threshold=threshold)

        method_tools["feature_list"] = feature_list
        method_tools["threshold"] = threshold
    
    elif opt.method == "fs-dgpm":
        postprocess_fsdgpm(opt=opt, model=model)

    elif opt.method in ["cclis", "cclis-bw", "cclis-wo", "cclis-wo-ss", "cclis-wo-is"]:

        method_tools = postprocess_cclis(opt, model, model2, method_tools, criterion, replay_indices)

    else:
        assert False

    return method_tools, model2