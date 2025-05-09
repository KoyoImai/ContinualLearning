
import numpy as np



def postprocess_fsdgpm(opt, model):

    # 閾値の獲得
    thres_value = min(opt.threshold + opt.target_task * opt.thres_add, opt.thres_last)
    thres = np.array([thres_value] * model.n_rep)

    print('-' * 60)
    print('Threshold: ', thres)

    # 特徴空間の更新（model.M_vecとmodel.M_valを更新）
    model.set_gpm_by_svd(thres)

