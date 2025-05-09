
export CUDA_VISIBLE_DEVICES="0"



python main_subspace.py --method co2l --dataset cifar10 \
                        --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/co2l_co2l_ring2000_cifar10_seed0_date2025_04_25/mem_log \
                        --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/co2l_co2l_ring2000_cifar10_seed0_date2025_04_25/model



python main_subspace.py --method co2l --dataset cifar100 \
                        --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/co2l_co2l_ring2000_cifar100_seed0_date2025_04_30/mem_log \
                        --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/co2l_co2l_ring2000_cifar100_seed0_date2025_04_30/model


