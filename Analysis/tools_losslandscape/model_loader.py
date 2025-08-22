import os
import torch





def load(dataset, model_name, model_file, net, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10_load(model_name, model_file, net, data_parallel)
    return net



def cifar10_load(model_name, model_file=None, net=None, data_parallel=False):
    if data_parallel: # the model is saved in data paralle mode
        net = torch.nn.DataParallel(net)

    # if model_file:
    #     assert os.path.exists(model_file), model_file + " does not exist."
    #     stored = torch.load(model_file, map_location=lambda storage, loc: storage)
    #     if 'state_dict' in stored.keys():
    #         net.load_state_dict(stored['state_dict'])
    #     else:
    #         net.load_state_dict(stored)


    if model_file:

        assert os.path.exists(model_file), model_file + " does not exist."
        stored = torch.load(model_file, map_location="cpu")

        # state_dictの読み出し
        state = stored.get("state_dict", stored.get("model", stored))
        # print("state: ", state)

        # DP使用時の "module" を取り除く
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
        
        # パラメータの読み込み
        net.load_state_dict(state, strict=False)


    if data_parallel: # convert the model back to the single GPU version
        net = net.module

    net.eval()
    return net