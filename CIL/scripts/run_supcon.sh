

export CUDA_VISIBLE_DEVICES="3"



python main.py --method supcon-joint --mem_type ring --dataset tiny-imagenet --batch_size 512 --epochs 1 --start_epoch 1 \
               --mem_size 2000 --seed 0 --log_name practice --date 2025_04_30 --linear_epochs 2