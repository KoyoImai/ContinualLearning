import os
import numpy as np

import torch

from util import AverageMeter



def score_computing(val_loader, model, model2, criterion, subset_sample_num, score_mask, opt):
    
    model.eval()
    max_iter = opt.max_iter

    # print("max_iter: ", max_iter)   # max_iter:  5
    
    for k, v in model.named_parameters():
        if k == 'prototypes.weight':
            print(k, v)
    
    losses = AverageMeter()
    distill = AverageMeter()

    cur_task_n_cls = (opt.target_task + 1)*opt.cls_per_task
    len_val_loader = sum(subset_sample_num)
    print('val_loader length', len_val_loader)

    all_score_sum = torch.zeros(cur_task_n_cls, cur_task_n_cls)
    _score = torch.zeros(cur_task_n_cls, len_val_loader)

    for i in range(max_iter):

        # listの初期化
        index_list, score_list, label_list = [], [], []
        
        # スコア合計の初期化
        score_sum = torch.zeros(cur_task_n_cls, cur_task_n_cls)
        # print("score_sum.shape: ", score_sum.shape)               # [クラス数, クラス数]

        for idx, (images, labels, importance_weight, index) in enumerate(val_loader):
            
            # インデックスリストとラベルリストを更新
            index_list += index
            label_list += labels
        
            # gpuに配置
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            # バッチサイズの獲得
            bsz = labels.shape[0]

            with torch.no_grad():

                # 過去タスクサンプルを取り出すためのマスク
                prev_task_mask = labels < opt.target_task * opt.cls_per_task
        
                # 特徴量と出力を獲得
                features, output = model(images)

                # 提案分布gの計算
                score_mat, batch_score_sum  = criterion.score_calculate(output, features, labels, importance_weight,index,
                                                                        target_labels=list(range(opt.target_task*opt.cls_per_task, (opt.target_task+1)*opt.cls_per_task)),
                                                                        sample_num = subset_sample_num, score_mask=score_mask)
                score_list.append(score_mat)

                score_sum += batch_score_sum

        index_list = torch.tensor(index_list) 
        label_list = torch.tensor(label_list).tolist()  

        mask = torch.eye(cur_task_n_cls)
        label_score_mask = torch.eq(torch.arange(cur_task_n_cls).view(-1, 1), torch.tensor(label_list)) 

        _score_list = torch.concat(score_list, dim=1) 
        _score_list = _score_list.to('cpu')

        _score -= _score * label_score_mask
        _score += (_score_list / _score_list.sum(dim=1, keepdim=True)) 
        all_score_sum += score_sum 
        all_score_sum -= all_score_sum * mask

    _score /= max_iter
    all_score_sum /= max_iter

    score_class_mask = None
    score = _score.cpu().sum(dim=0) / (_score.shape[0] - 1)

        
    return score_class_mask, index_list, score, model2





def postprocess_cclis(opt, model, model2, method_tools, criterion, replay_indices):

    score_mask = method_tools["score_mask"]
    val_loader = method_tools["post_loader"]
    subset_sample_num = method_tools["subset_sample_num"]
    val_targets = method_tools["val_targets"]

    # print("score_mask: ", score_mask)
    score_mask, index, _score, model2 = score_computing(val_loader, model, model2, criterion, subset_sample_num, score_mask, opt)  # compute score

    print(opt.target_task)
    observed_classes = list(range(opt.target_task * opt.cls_per_task, (opt.target_task + 1) * opt.cls_per_task))

    observed_indices = []
    for tc in observed_classes:
        observed_indices += np.where(val_targets == tc)[0].tolist()

    print('replay_indices_len', len(replay_indices))
    print('observed_indices_len', len(observed_indices))
    score_indices = replay_indices + observed_indices

    score_dict = dict(zip(np.array(index), _score))
    score = torch.stack([score_dict[key] for key in score_indices])
    print('score', score)

    # save the last score
    np.save(
        os.path.join(opt.mem_path, 'score_{target_task}.npy'.format(target_task=opt.target_task)),
        np.array(score.cpu()))


    method_tools["score"] = score
    method_tools["score_mask"] = score_mask

    return method_tools
    
