
export CUDA_VISIBLE_DEVICES="3"





python main.py --method co2l --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 0.5 --mem_size 500 --current_temp 0.2 --past_temp 0.01 --distill_power 0.0 \
               --seed 0 --log_name co2l_joint --date 2025_05_10

