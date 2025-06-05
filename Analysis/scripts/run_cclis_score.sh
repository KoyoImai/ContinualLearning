
export CUDA_VISIBLE_DEVICES="2,3"

# python main_score.py --method cclis --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/model




# python main_score.py --method cclis --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed1_date2025_05_11/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed1_date2025_05_11/model


# python main_score.py --method cclis --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_05_12/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_05_12/model






# CCLIS w/o distillation
# python main_score.py --method cclis --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed0_date2025_05_21/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed0_date2025_05_21/model

# python main_score.py --method cclis --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed1_date2025_05_21/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed1_date2025_05_21/model

# python main_score.py --method cclis --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed2_date2025_05_21/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed2_date2025_05_21/model

# python main_score.py --method cclis --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed3_date2025_05_21/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed3_date2025_05_21/model

# python main_score.py --method cclis --dataset cifar10 \
#                      --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed4_date2025_05_21/mem_log \
#                      --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed4_date2025_05_21/model


# python main_score.py --method cclis --dataset cifar100 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed0_date2025_05_21/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed0_date2025_05_21/model

# python main_score.py --method cclis --dataset cifar100 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed1_date2025_05_21/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed1_date2025_05_21/model

# python main_score.py --method cclis --dataset cifar100 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed2_date2025_05_21/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed2_date2025_05_21/model

# python main_score.py --method cclis --dataset cifar100 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed3_date2025_05_21/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed3_date2025_05_21/model

# python main_score.py --method cclis --dataset cifar100 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed4_date2025_05_21/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed4_date2025_05_21/model






# 通常のCCLIS（tiny-imagenet）
python main_score.py --method cclis --dataset tiny-imagenet --use_dp\
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed0_date2025_05_30/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed0_date2025_05_30/model

# python main_score.py --method cclis --dataset tiny-imagenet --use_dp \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed1_date2025_05_30/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed1_date2025_05_30/model

# python main_score.py --method cclis --dataset tiny-imagenet --use_dp \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed2_date2025_05_30/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed2_date2025_05_30/model

# python main_score.py --method cclis --dataset tiny-imagenet --use_dp \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed3_date2025_05_30/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed3_date2025_05_30/model

# python main_score.py --method cclis --dataset tiny-imagenet --use_dp \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed4_date2025_05_30/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed4_date2025_05_30/model









