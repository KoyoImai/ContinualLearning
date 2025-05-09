
export CUDA_VISIBLE_DEVICES="0"



python main_subspace.py --method cclis --dataset cifar10 \
                        --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_04_20/mem_log \
                        --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_04_20/model



# python main_subspace.py --method cclis --dataset cifar100 \
#                         --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar100_seed0_date2025_04_20/mem_log \
#                         --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar100_seed0_date2025_04_20/model






