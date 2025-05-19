


export CUDA_VISIBLE_DEVICES="1"





# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_04_20/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_04_20/model


# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed1_date2025_04_20/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed1_date2025_04_20/model


# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed2_date2025_04_20/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed2_date2025_04_20/model


# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed3_date2025_04_20/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed3_date2025_04_20/model


# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed4_date2025_04_20/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed4_date2025_04_20/model






# /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_05_12


## ここから分析ように実行したやつ
# python main_svd.py --method cclis --dataset cifar10 --block_type block --flatten_type avgpool \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed1_date2025_05_11/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed1_date2025_05_11/model

python main_svd.py --method cclis --dataset cifar10 --block_type block --flatten_type avgpool \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_05_12/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_05_12/model



# python main_svd.py --method cclis --dataset cifar10 --block_type conv --flatten_type flatten \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_04_20/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_04_20/model

# python main_svd.py --method cclis --dataset cifar10 --projector\
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_04_20/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_04_20/model



# python main_svd.py --method cclis --dataset cifar100 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar100_seed0_date2025_04_20/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar100_seed0_date2025_04_20/model


