export CUDA_VISIBLE_DEVICES="2"





# cifar10
##############################################################################################################################################################################
# # SupCon-joint
# python main_score.py --method supcon-joint --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/2025_0512/lr1_supcon-joint_ring0_cifar10_seed1_date2025_05_12/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/2025_0512/lr1_supcon-joint_ring0_cifar10_seed1_date2025_05_12/model


# # # CCLIS
# python main_score.py --method cclis --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed1_date2025_05_11/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed1_date2025_05_11/model


# # Co2L
# python main_score.py --method co2l --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/2025_0513/co2l_co2l_ring2000_cifar10_seed1_date2025_05_13/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/2025_0513/co2l_co2l_ring2000_cifar10_seed1_date2025_05_13/model


# # SupCon（メモリ0）
# python main_score.py --method supcon-joint --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/2025_0509/practice_supcon_ring0_cifar10_seed0_date2025_05_09/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/2025_0509/practice_supcon_ring0_cifar10_seed0_date2025_05_09/model




# cifaf100
##############################################################################################################################################################################
# SupCon-joint
python main_score.py --method supcon-joint --dataset cifar100 \
                     --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/2025_0512/lr1_supcon-joint_ring0_cifar100_seed1_date2025_05_12/mem_log \
                     --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/2025_0512/lr1_supcon-joint_ring0_cifar100_seed1_date2025_05_12/model


# # CCLIS
python main_score.py --method cclis --dataset cifar100 \
                     --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar100_seed1_date2025_05_11/mem_log \
                     --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar100_seed1_date2025_05_11/model


# Co2L
python main_score.py --method co2l --dataset cifar100 \
                     --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/2025_0513/co2l_co2l_ring2000_cifar100_seed1_date2025_05_13/mem_log \
                     --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/2025_0513/co2l_co2l_ring2000_cifar100_seed1_date2025_05_13/model




