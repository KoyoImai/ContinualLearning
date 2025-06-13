
export CUDA_VISIBLE_DEVICES="3"

# # # CCLIS
# python main_umap_epoch.py --method cclis --dataset cifar10 \
#                           --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0606/default/cclis_cclis_ring2000_cifar10_seed0_date2025_06_06/mem_log \
#                           --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0606/default/cclis_cclis_ring2000_cifar10_seed0_date2025_06_06/model



# # CCLIS
python main_umap_epoch_v2.py --method cclis --dataset cifar10 \
                          --log_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0606/default/cclis_cclis_ring2000_cifar10_seed0_date2025_06_06/mem_log \
                          --model_path /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/2025_0606/default/cclis_cclis_ring2000_cifar10_seed0_date2025_06_06/model




