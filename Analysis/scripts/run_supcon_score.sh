export CUDA_VISIBLE_DEVICES="2"



python main_score.py --method supcon-joint --dataset cifar10  \
                     --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/2025_0512/lr1_supcon-joint_ring0_cifar10_seed1_date2025_05_12/mem_log \
                     --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon-joint/2025_0512/lr1_supcon-joint_ring0_cifar10_seed1_date2025_05_12/model


