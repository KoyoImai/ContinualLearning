
import numpy as np

import torch


def preprocess_gpm(opt, method_tools):

    # 特徴表現行列の計算
    feature_mat = []
    feature_list = method_tools["feature_list"]
    for i in range(len(feature_list)):
        if torch.cuda.is_available():
            Uf=torch.Tensor(np.dot(feature_list[i],feature_list[i].transpose())).cuda()
        print('Layer {} - Projection Matrix shape: {}'.format(i+1,Uf.shape))
        feature_mat.append(Uf)
    print ('-'*40)

    method_tools["feature_mat"] = feature_mat

    return method_tools