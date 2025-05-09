


export CUDA_VISIBLE_DEVICES="1"




python main.py --method cclis --mem_type ring --dataset tiny-imagenet --batch_size 512 --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --cosine \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 50 --start_epoch 500 --log_name cclis --date 2025_05_01

python main.py --method cclis --mem_type ring --dataset tiny-imagenet --batch_size 512 --seed 1 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --cosine \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 50 --start_epoch 500 --log_name cclis --date 2025_05_01

python main.py --method cclis --mem_type ring --dataset tiny-imagenet --batch_size 512 --seed 2 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --cosine \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 50 --start_epoch 500 --log_name cclis --date 2025_05_01

python main.py --method cclis --mem_type ring --dataset tiny-imagenet --batch_size 512 --seed 3 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --cosine \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 50 --start_epoch 500 --log_name cclis --date 2025_05_01

python main.py --method cclis --mem_type ring --dataset tiny-imagenet --batch_size 512 --seed 4 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --cosine \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 50 --start_epoch 500 --log_name cclis --date 2025_05_01



