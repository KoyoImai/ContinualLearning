


export CUDA_VISIBLE_DEVICES="2"



python main.py --method co2l --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --mem_size 2000 --current_temp 0.2 --past_temp 0.01 --distill_power 1.0 \
               --seed 0 --log_name co2l --date 2025_04_25

python main.py --method co2l --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --mem_size 2000 --current_temp 0.2 --past_temp 0.01 --distill_power 1.0 \
               --seed 1 --log_name co2l --date 2025_04_25

python main.py --method co2l --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --mem_size 2000 --current_temp 0.2 --past_temp 0.01 --distill_power 1.0 \
               --seed 2 --log_name co2l --date 2025_04_25

python main.py --method co2l --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --mem_size 2000 --current_temp 0.2 --past_temp 0.01 --distill_power 1.0 \
               --seed 3 --log_name co2l --date 2025_04_25

python main.py --method co2l --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --mem_size 2000 --current_temp 0.2 --past_temp 0.01 --distill_power 1.0 \
               --seed 4 --log_name co2l --date 2025_04_25
