


from preprocesses.preprocess_cclis import preprocess_cclis
from preprocesses.preprocess_prco import preprocess_prco


def pre_process(opt, model, model2, dataloader, method_tools):

    if opt.method in ["co2l"]:
        return method_tools, model, model2

    elif opt.method in ["cclis"]:

        preprocess_cclis(opt, model, method_tools)

        return method_tools, model, model2
    
    elif opt.method in ["prco"]:

        preprocess_prco(opt, model, method_tools)

        return method_tools, model, model2







