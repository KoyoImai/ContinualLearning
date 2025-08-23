



# export original_path="/home/kouyou/ContinualLearning/survey/CIL/logs/cclis/debug_cclis_ring500_cifar10_seed0_date2025_0822"
export original_path="/home/kouyou/task00"


# cclis デバッグ
# ランダム方向へ摂動を加え，3次元で可視化を行う場合，
# python main_losslandscape.py \
#        --method cclis \
#        --cls_list 0 1 \
#        --model_file ${original_path}/model/task00/model_epoch001.pth \
#        --classifier_file ${original_path}/model/task00/classifier_epoch001.pth \
#        --x=-1:1:51 --y=-1:1:51 \
#        --surf_file ./use_classifier.h5 \
#        --use_classifier \
#        --plot \
#        --dir_type weights

python main_losslandscape.py \
       --method cclis \
       --cls_list 0 1 \
       --model_file ${original_path}/model_epoch150.pth \
       --classifier_file ${original_path}/classifier_epoch150.pth \
       --x=-1:1:51 --y=-1:1:51 \
       --surf_file ./use_classifier.h5 \
       --use_classifier \
       --plot \
       --dir_type weights


