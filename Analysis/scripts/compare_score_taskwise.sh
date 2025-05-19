export CUDA_VISIBLE_DEVICES="2"





# cifar10
##############################################################################################################################################################################
# SupCon-joint
# python main_score_taskwise.py --method supcon-joint --dataset cifar10 \
#                               --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/2025_0501/practice_supcon-joint_ring0_cifar10_seed1_date2025_05_01/mem_log \
#                               --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/2025_0501/practice_supcon-joint_ring0_cifar10_seed1_date2025_05_01/model


# # CCLIS
# python main_score_taskwise.py --method cclis --dataset cifar10 \
#                               --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/mem_log \
#                               --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/model


# # SupCon（メモリ0）
# python main_score_taskwise.py --method supcon-joint --dataset cifar10 \
#                               --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/2025_0509/practice_supcon_ring0_cifar10_seed0_date2025_05_09/mem_log \
#                               --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/2025_0509/practice_supcon_ring0_cifar10_seed0_date2025_05_09/model

        

# cifaf100
##############################################################################################################################################################################
python main_score_taskwise.py --method supcon-joint --dataset cifar100 \
                              --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/2025_0501/practice_supcon-joint_ring0_cifar100_seed0_date2025_05_01/mem_log \
                              --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/2025_0501/practice_supcon-joint_ring0_cifar100_seed0_date2025_05_01/model


# CCLIS
python main_score_taskwise.py --method cclis --dataset cifar100 \
                              --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar100_seed0_date2025_05_11/mem_log \
                              --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar100_seed0_date2025_05_11/model


# SupCon（メモリ0）
python main_score_taskwise.py --method supcon-joint --dataset cifar100 \
                              --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/2025_0509/practice_supcon_ring0_cifar100_seed0_date2025_05_09/mem_log \
                              --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/2025_0509/practice_supcon_ring0_cifar100_seed0_date2025_05_09/model

