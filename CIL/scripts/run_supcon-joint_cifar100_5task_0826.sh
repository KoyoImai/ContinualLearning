export CUDA_VISIBLE_DEVICES="1"



python ./main.py --method supcon-joint --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --temp 0.1 --mem_size 0 --seed 0 --epoch_save --log_name supcon-joint-default --date 2025_0826