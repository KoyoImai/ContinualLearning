

from extract.extract_cclis import extract_features_cclis, extract_features_cclis_projector
from extract.extract_er import extract_features_er


def extract_features(opt, model, data_loader):


    if opt.method in ["cclis", "co2l", "supcon-joint"]:
        # print("model: ", model)
        # assert False
        if opt.projector:
            features, labels = extract_features_cclis_projector(opt=opt, model=model, data_loader=data_loader)
            layer_outputs = None
        else:
            # if opt.use_dp:
            #     features, labels, layer_outputs = extract_features_cclis(opt=opt, model=model.module, data_loader=data_loader)
            # else:
            features, labels, layer_outputs = extract_features_cclis(opt=opt, model=model, data_loader=data_loader)

    elif opt.method in ["er"]:

        features, labels = extract_features_er(opt=opt, model=model, data_loader=data_loader)
        ayer_outputs = None
    
    else:
        assert False
    

    return features, labels, layer_outputs



