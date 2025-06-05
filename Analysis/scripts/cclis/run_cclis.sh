


export CUDA_VISIBLE_DEVICES="3"





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

# python main_svd.py --method cclis --dataset cifar10 --block_type block --flatten_type avgpool \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_05_12/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_05_12/model



# python main_svd.py --method cclis --dataset cifar10 --block_type conv --flatten_type flatten \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_04_20/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_04_20/model

# python main_svd.py --method cclis --dataset cifar10 --projector\
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_04_20/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar10_seed0_date2025_04_20/model



# python main_svd.py --method cclis --dataset cifar100 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar100_seed0_date2025_04_20/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/cclis_cclis_ring2000_cifar100_seed0_date2025_04_20/model







# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/model


# # CCLIS-RFR
# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-01_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-01_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/model

# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-02_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-02_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/model

# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-03_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-03_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/model

# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-04_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-04_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/model

# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-05_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-05_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/model

# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-06_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-06_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/model

# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-07_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-07_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/model

# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-08_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-08_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/model

# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-09_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-09_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/model

# python main_svd.py --method cclis --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-1_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-rfr/2025_0522/cclis-rfr-1_cclis-rfr_ring2000_cifar10_seed0_date2025_05_22/model





# 通常のCCLIS（tiny-imagenet）
python main_svd.py --method cclis --dataset tiny-imagenet \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed0_date2025_05_30/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed0_date2025_05_30/model

python main_svd.py --method cclis --dataset tiny-imagenet \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed1_date2025_05_30/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed1_date2025_05_30/model

python main_svd.py --method cclis --dataset tiny-imagenet \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed2_date2025_05_30/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed2_date2025_05_30/model

python main_svd.py --method cclis --dataset tiny-imagenet \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed3_date2025_05_30/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed3_date2025_05_30/model

python main_svd.py --method cclis --dataset tiny-imagenet \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed4_date2025_05_30/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0530/cclis_cclis_ring2000_tiny-imagenet_seed4_date2025_05_30/model





# CCLIS w/o distillation
python main_svd.py --method cclis --dataset cifar10 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed0_date2025_05_21/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed0_date2025_05_21/model

python main_svd.py --method cclis --dataset cifar10 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed1_date2025_05_21/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed1_date2025_05_21/model

python main_svd.py --method cclis --dataset cifar10 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed2_date2025_05_21/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed2_date2025_05_21/model

python main_svd.py --method cclis --dataset cifar10 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed3_date2025_05_21/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed3_date2025_05_21/model

python main_svd.py --method cclis --dataset cifar10 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed4_date2025_05_21/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar10_seed4_date2025_05_21/model


python main_svd.py --method cclis --dataset cifar100 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed0_date2025_05_21/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed0_date2025_05_21/model

python main_svd.py --method cclis --dataset cifar100 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed1_date2025_05_21/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed1_date2025_05_21/model

python main_svd.py --method cclis --dataset cifar100 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed2_date2025_05_21/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed2_date2025_05_21/model

python main_svd.py --method cclis --dataset cifar100 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed3_date2025_05_21/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed3_date2025_05_21/model

python main_svd.py --method cclis --dataset cifar100 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed4_date2025_05_21/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0521/cclis_cclis_ring2000_cifar100_seed4_date2025_05_21/model









