


from preprocesses.preprocess_cclis import preprocess_cclis
from preprocesses.preprocess_prco import preprocess_prco


def pre_process(opt, model, model2, dataloader, method_tools):

    if opt.method in ["co2l"]:
        return method_tools, model, model2

    elif opt.method in ["cclis"]:

        preprocess_cclis(opt, model, method_tools)

        return method_tools, model, model2
    
    elif opt.method in ["prco", "prco-fimcl", "prco-fimclv2", "prco-fimclv3", "prco-efm"]:

        preprocess_prco(opt, model, method_tools)

        return method_tools, model, model2

    elif opt.method in ["prco-progefm"]:

        preprocess_prco(opt, model, method_tools)

        return method_tools, model, model2
    
    elif opt.method in ["prco-ema"]:

        preprocess_prco(opt, model, method_tools)

        if (opt.target_task == 0) or opt.emamodel_reset:
            model.module.reset_ema_model()
        else:
            assert False
        return method_tools, model, model2


    # reset_ema_model

    else:
        assert False





