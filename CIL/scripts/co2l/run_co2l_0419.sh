
export CUDA_VISIBLE_DEVICES="0"

python main.py --method co2l --mem_type ring --dataset tiny-imagenet --batch_size 512 --epochs 100 --start_epoch 500 \
               --mem_size 500 --current_temp 0.1 --past_temp 0.1 --temp 0.5 --distill_power 1.0 \
               --seed 0 --log_name co2l --date 2025_04_19

python main.py --method co2l --mem_type ring --dataset tiny-imagenet --batch_size 512 --epochs 100 --start_epoch 500 \
               --mem_size 500 --current_temp 0.1 --past_temp 0.1 --temp 0.5 --distill_power 1.0 \
               --seed 1 --log_name co2l --date 2025_04_19

python main.py --method co2l --mem_type ring --dataset tiny-imagenet --batch_size 512 --epochs 100 --start_epoch 500 \
               --mem_size 500 --current_temp 0.1 --past_temp 0.1 --temp 0.5 --distill_power 1.0 \
               --seed 2 --log_name co2l --date 2025_04_19

python main.py --method co2l --mem_type ring --dataset tiny-imagenet --batch_size 512 --epochs 100 --start_epoch 500 \
               --mem_size 500 --current_temp 0.1 --past_temp 0.1 --temp 0.5 --distill_power 1.0 \
               --seed 3 --log_name co2l --date 2025_04_19

python main.py --method co2l --mem_type ring --dataset tiny-imagenet --batch_size 512 --epochs 100 --start_epoch 500 \
               --mem_size 500 --current_temp 0.1 --past_temp 0.1 --temp 0.5 --distill_power 1.0 \
               --seed 4 --log_name co2l --date 2025_04_19