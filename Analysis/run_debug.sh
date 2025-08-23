


export original_path="/home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0810/cclis_graddetail_cclis_ring500_cifar10_seed0_date2025_0810"


# cclis デバッグ
# # ランダム方向へ摂動を加え，3次元で可視化を行う場合，
# python main_losslandscape.py \
#        --method cclis \
#        --cls_list 0 1 2 3 \
#        --model_file ${original_path}/model/model_02.pth \
#        --x=-1:1:51 --y=-1:1:51 \
#        --surf_file /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0810/cclis_graddetail_cclis_ring500_cifar10_seed0_date2025_0810/losslandscape/issupcon_task1.h5 \
#        --plot \
#        --dir_type weights

# python h52vtp.py --surf_file ./losslandscape_cclis.h5 --surf_name train_loss --zmax  10 --log


# --model_file2 を与える場合
python main_losslandscape.py \
       --method cclis \
       --cls_list 0 1 2 3 \
       --model_file ${original_path}/model/model_00.pth \
       --model_file2 ${original_path}/model/task01/model_epoch050.pth \
       --model_file3 ${original_path}/model/model_01.pth \
       --x=-1:1:51 --y=-1:1:51 \
       --surf_file ./losslandscape_3model.h5 \
       --plot \
       --dir_type weights




# ## co2l デバッグ
# python main_losslandscape.py \
#        --method co2l \
#        --cls_list 0 1 2 3 \
#        --model_file /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/2025_0602/co2l_co2l_ring500_tiny-imagenet_seed0_date2025_06_02/model/model_01.pth
    










