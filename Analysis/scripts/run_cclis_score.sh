
export CUDA_VISIBLE_DEVICES="2"

# python main_score.py --method cclis --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/model




python main_score.py --method cclis --dataset cifar10 \
                     --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed1_date2025_05_11/mem_log \
                     --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed1_date2025_05_11/model




python main_score.py --method cclis --dataset cifar10 \
                     --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_05_12/mem_log \
                     --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_05_12/model
