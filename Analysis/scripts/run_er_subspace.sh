
export CUDA_VISIBLE_DEVICES="0"



python main_subspace.py --method er --dataset cifar10 \
                        --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/er/erring_er_ring2000_cifar10_seed0_date2025_04_19/mem_log \
                        --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/er/erring_er_ring2000_cifar10_seed0_date2025_04_19/model


