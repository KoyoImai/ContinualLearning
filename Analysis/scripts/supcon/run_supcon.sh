

export CUDA_VISIBLE_DEVICES="2"



# python main_svd.py --method supcon-joint --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/practice_supcon-joint_ring0_cifar10_seed1_date2025_05_01/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/practice_supcon-joint_ring0_cifar10_seed1_date2025_05_01/model

# 学種率 1.0 cifar10
# python main_svd.py --method supcon-joint --dataset cifar10 --block_type block --flatten_type avgpool \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0512/lr1_supcon-joint_ring0_cifar10_seed0_date2025_05_12/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0512/lr1_supcon-joint_ring0_cifar10_seed0_date2025_05_12/model

# 学習率0.02 cifar10
# python main_svd.py --method supcon-joint --dataset cifar10 --block_type block --flatten_type avgpool \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0512/lr002_supcon-joint_ring0_cifar10_seed0_date2025_05_12/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0512/lr002_supcon-joint_ring0_cifar10_seed0_date2025_05_12/model

# オフライン cifar10
# python main_svd.py --method supcon-joint --dataset cifar10 --block_type block --flatten_type avgpool \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/offline_supcon_ring0_cifar10_seed0_date2025_05_13/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/offline_supcon_ring0_cifar10_seed0_date2025_05_13/model

#オフライン cifar100
python main_svd.py --method supcon-joint --dataset cifar100 --block_type block --flatten_type avgpool \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/offline_supcon_ring0_cifar10_seed0_date2025_05_13/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/offline_supcon_ring0_cifar10_seed0_date2025_05_13/model


# /exp_log


# python main_svd.py --method supcon-joint --dataset cifar100 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/practice_supcon-joint_ring0_cifar100_seed0_date2025_05_01/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/practice_supcon-joint_ring0_cifar100_seed0_date2025_05_01/model




