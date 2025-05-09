
export CUDA_VISIBLE_DEVICES="0"



python main_subspace.py --method supcon-joint --dataset cifar10 \
                        --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/practice_supcon-joint_ring0_cifar10_seed1_date2025_05_01/mem_log \
                        --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/practice_supcon-joint_ring0_cifar10_seed1_date2025_05_01/model




# python main_subspace.py --method supcon-joint --dataset cifar100 \
#                         --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/practice_supcon-joint_ring0_cifar100_seed0_date2025_05_01/mem_log \
#                         --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/practice_supcon-joint_ring0_cifar100_seed0_date2025_05_01/model








