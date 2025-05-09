


from scipy import optimize
import torch.optim as optim



def preprocess_fsdgpm(opt, model, method_tools):

    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    model.eta1 = opt.eta1

    if len(model.M_vec) > 0 and opt.fsdgpm_method in ['dgpm', 'xdgpm']:

        model.eta2 = opt.eta2
        model.define_lambda_params()
        model.update_opt_lambda(model.eta2)


    method_tools['optimizer'] = optimizer

    return model, method_tools