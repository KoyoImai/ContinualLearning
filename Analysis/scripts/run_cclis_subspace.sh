
export CUDA_VISIBLE_DEVICES="2"



# python main_subspace.py --method cclis --dataset cifar10 \
#                         --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/mem_log \
#                         --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/model


python main_subspace.py --method cclis --dataset cifar10 --rank 20 \
                        --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/mem_log \
                        --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/model



# python main_subspace.py --method cclis --dataset cifar100 \
#                         --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar100_seed0_date2025_04_20/mem_log \
#                         --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar100_seed0_date2025_04_20/model






