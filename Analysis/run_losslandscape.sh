

export original_path="/home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0810/cclis_graddetail_cclis_ring500_cifar10_seed0_date2025_0810"

# ランダム方向へ摂動を加え，3次元で可視化を行う場合，
python main_losslandscape.py \
       --method cclis \
       --cls_list 0 1 2 3 \
       --model_file ${original_path}/model/model_02.pth \
       --x=-1:1:51 --y=-1:1:51 \
       --surf_file /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0810/cclis_graddetail_cclis_ring500_cifar10_seed0_date2025_0810/losslandscape/issupcon_task1_class0123.h5 \
       --plot \
       --dir_type weights




