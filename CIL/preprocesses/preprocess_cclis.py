import math


def preprocess_cclis(opt, model, method_tools):

    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        # opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from_enc = 0.01
        opt.warmup_from_prot = 0.001
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min_encoder = opt.learning_rate * (opt.lr_decay_rate ** 3)
            eta_min_prototypes = opt.learning_rate_prototypes * (opt.lr_decay_rate ** 3)
            opt.warmup_to_enc = eta_min_encoder + (opt.learning_rate - eta_min_encoder) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
            opt.warmup_to_prot = eta_min_prototypes + (opt.learning_rate_prototypes - eta_min_prototypes) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to_enc = opt.learning_rate
            opt.warmup_to_prot = opt.learning_rate_prototypes