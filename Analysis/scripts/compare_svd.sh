

export CUDA_VISIBLE_DEVICES="2"





# CIFAR10
###############################################################################################################################################################################
# CCLIS
python main_svd.py --method cclis --dataset cifar10 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar10_seed0_date2025_05_11/model

# Co2L
python main_svd.py --method co2l --dataset cifar10 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/2025_0513/co2l_co2l_ring2000_cifar10_seed0_date2025_05_13/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/2025_0513/co2l_co2l_ring2000_cifar10_seed0_date2025_05_13/model

# SupCon-joint
python main_svd.py --method supcon-joint --dataset cifar10 \
                   --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0512/lr1_supcon-joint_ring0_cifar10_seed0_date2025_05_12/mem_log \
                   --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0512/lr1_supcon-joint_ring0_cifar10_seed0_date2025_05_12/model

# SupCon (メモリ０)
# クラスタで実行中
# python main_svd.py --method supcon-joint --dataset cifar10 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/2025_0508/practice_supcon_ring0_cifar10_seed0_date2025_05_08/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/2025_0508/practice_supcon_ring0_cifar10_seed0_date2025_05_08/model

# SupCon (メモリ2000)
# クラスタで実行中







# CIFAR10
###############################################################################################################################################################################
# # CCLIS
# python main_svd.py --method cclis --dataset cifar100 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar100_seed0_date2025_05_11/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0511/cclis_cclis_ring2000_cifar100_seed0_date2025_05_11/model

# # Co2L
# python main_svd.py --method co2l --dataset cifar100 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/2025_0513/co2l_co2l_ring2000_cifar100_seed0_date2025_05_13/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/co2l/2025_0513/co2l_co2l_ring2000_cifar100_seed0_date2025_05_13/model

# # SupCon-joint
# python main_svd.py --method supcon-joint --dataset cifar100 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0512/lr1_supcon-joint_ring0_cifar100_seed0_date2025_05_12/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0512/lr1_supcon-joint_ring0_cifar100_seed0_date2025_05_12/model

# SupCon (メモリ０)
# python main_svd.py --method supcon-joint --dataset cifar100 \
#                    --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/2025_0508/practice_supcon_ring0_cifar100_seed0_date2025_05_08/mem_log \
#                    --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/supcon/2025_0508/practice_supcon_ring0_cifar100_seed0_date2025_05_08/model


