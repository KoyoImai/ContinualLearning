
import torch


from preprocesses.preprocess_gpm import preprocess_gpm
from preprocesses.preprocess_lucir import preprocess_lucir
from preprocesses.preprocess_fsdgpm import preprocess_fsdgpm
from preprocesses.preprocess_cclis import preprocess_cclis



def pre_process(opt, model, model2,  dataloader, method_tools):

    if opt.method in ["co2l", "supcon", "supcon-joint", "simclr"]:
        return method_tools, model, model2
    elif opt.method in ["er"]:
        if opt.target_task == 0:
            print("no process")
        else:
            
            # fc層を追加
            new_params = model.update_fc()
            if torch.cuda.is_available():
                model = model.cuda()

            optimizer = method_tools["optimizer"]

            # print("optimizer.param_groups: ", optimizer.param_groups)
            # print("optimizer.param_groups.keys(): ", optimizer.param_groups.keys())

            # # optimizer の momentum を確認
            # ref_p = next(iter(optimizer.state))  # 既存パラメータの1つ
            # print("ref_p: ", ref_p)
            # m_before = optimizer.state[ref_p]['momentum_buffer'].clone()
            # print("m_before: ", m_before)
            
            # 追加したfc層をoptimizerの最適化対象に加える
            optimizer.add_param_group({
                "params": new_params,
                "lr": opt.learning_rate,
                "momentum": opt.momentum,
                "weight_decay": opt.weight_decay,
            })
            optimizer.zero_grad(set_to_none=True)

            # print("Δmomentum(max abs):",
            #         (optimizer.state[ref_p]['momentum_buffer'] - m_before).abs().max().item())


    elif opt.method == "gpm":
        method_tools = preprocess_gpm(opt=opt, method_tools=method_tools)
    elif opt.method == "lucir":
        method_tools, model, model2 = preprocess_lucir(opt=opt, model=model, model2=model2, method_tools=method_tools)
        return method_tools, model, model2
    elif opt.method in ["fs-dgpm"]:
        model, method_tools = preprocess_fsdgpm(opt, model, method_tools)
        return method_tools, model, model2
    elif opt.method in ["cclis", "cclis-wo", "cclis-bw", "cclis-rfr", "cclis-pcgrad"]:
        preprocess_cclis(opt, model, method_tools)
        # print("opt.warm: ", opt.warm)
        # assert False

    elif opt.method in ["cclis-wo-ss", "cclis-wo-is"]:
        return method_tools, model, model2
    else:
        assert False

    return method_tools, model, model2