
export CUDA_VISIBLE_DEVICES="2"



python main_subspace.py --method supcon-joint --dataset cifar10 --rank 20 \
                        --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/2025_0512/lr1_supcon-joint_ring0_cifar10_seed0_date2025_05_12/mem_log \
                        --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/2025_0512/lr1_supcon-joint_ring0_cifar10_seed0_date2025_05_12/model




# python main_subspace.py --method supcon-joint --dataset cifar100 \
#                         --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/practice_supcon-joint_ring0_cifar100_seed0_date2025_05_01/mem_log \
#                         --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/practice_supcon-joint_ring0_cifar100_seed0_date2025_05_01/model








