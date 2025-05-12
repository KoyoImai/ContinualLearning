


def set_buffer(opt, model, prev_indices=None, method_tools=None):

    # 手法毎にバッファの作成方法を変更
    if opt.method == "er":
        from dataloaders.buffer_er import set_replay_samples_reservoir
        from dataloaders.buffer_er import set_replay_samples_ring

        if opt.mem_type == "reservoir":
            replay_indices = set_replay_samples_reservoir(opt, model, prev_indices=prev_indices)
        elif opt.mem_type == "ring":
            replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices)
        else:
            assert False


    elif opt.method == "co2l":

        if opt.mem_type == "ring":
            from dataloaders.buffer_er import set_replay_samples_ring
            replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices)
        elif opt.mem_type == "herding":
            from dataloaders.buffer_lucir import set_replay_samples_herding
            replay_indices = set_replay_samples_herding(opt, model, prev_indices=prev_indices)
    

    elif opt.method == "gpm":
        from dataloaders.buffer_er import set_replay_samples_reservoir
        from dataloaders.buffer_er import set_replay_samples_ring

        if opt.mem_type == "reservoir":
            replay_indices = set_replay_samples_reservoir(opt, model, prev_indices=prev_indices)
        elif opt.mem_type == "ring":
            replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices)
        else:
            assert False


    elif opt.method == "lucir":
        from dataloaders.buffer_er import set_replay_samples_reservoir
        from dataloaders.buffer_er import set_replay_samples_ring
        from dataloaders.buffer_lucir import set_replay_samples_herding
        if opt.mem_type == "reservoir":
            replay_indices = set_replay_samples_reservoir(opt, model, prev_indices=prev_indices)
        elif opt.mem_type == "ring":
            replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices)
        elif opt.mem_type == "herding":
            replay_indices = set_replay_samples_herding(opt, model, prev_indices=prev_indices)
        else:
            assert False


    elif opt.method == "fs-dgpm":
        replay_indices = []


    elif opt.method in ["cclis", "cclis-wo", "cclis-bw"]:

        from dataloaders.buffer_cclis import set_replay_samples_cclis
        
        importance_weight = method_tools["importance_weight"]
        score = method_tools["score"]

        replay_indices, importance_weight, val_targets = set_replay_samples_cclis(
            opt, prev_indices=prev_indices, prev_importance_weight=importance_weight, prev_score=score
        )  # [prev_sample_num] tensor
        print("replauy_indices: ", replay_indices)
        # print("importance_weight: ", importance_weight)
        # print("val_targets: ", val_targets)
        # print("len(val_targets): ", len(val_targets))

        method_tools["importance_weight"] = importance_weight
        method_tools["val_targets"] = val_targets
    
    elif opt.method in ['supcon']:
        if opt.mem_type == "ring":
            from dataloaders.buffer_er import set_replay_samples_ring
            replay_indices = set_replay_samples_ring(opt, model, prev_indices=prev_indices)
        elif opt.mem_type == "herding":
            from dataloaders.buffer_lucir import set_replay_samples_herding
            replay_indices = set_replay_samples_herding(opt, model, prev_indices=prev_indices)
    
    elif opt.method in ["cclis-wo-ss"]:

        from dataloaders.buffer_cclis_wo_ss import set_replay_samples_cclis_wo_ss

        importance_weight = method_tools["importance_weight"]
        score = method_tools["score"]

        replay_indices, importance_weight, val_targets = set_replay_samples_cclis_wo_ss(
            opt, prev_indices=prev_indices, prev_importance_weight=importance_weight, prev_score=score
        )  # [prev_sample_num] tensor
        print("replauy_indices: ", replay_indices)

        method_tools["importance_weight"] = importance_weight
        method_tools["val_targets"] = val_targets
    

    elif opt.method in ["cclis-wo-is"]:

        from dataloaders.buffer_cclis_wo_is import set_replay_samples_cclis_wo_is

        importance_weight = method_tools["importance_weight"]
        score = method_tools["score"]

        replay_indices, importance_weight, val_targets = set_replay_samples_cclis_wo_is(
            opt, prev_indices=prev_indices, prev_importance_weight=importance_weight, prev_score=score
        )  # [prev_sample_num] tensor
        print("replauy_indices: ", replay_indices)

        method_tools["importance_weight"] = importance_weight
        method_tools["val_targets"] = val_targets

    
    elif opt.method in ['supcon-joint']:

        replay_indices = []
        
    else:
        assert False
    

    


    return replay_indices, method_tools






    