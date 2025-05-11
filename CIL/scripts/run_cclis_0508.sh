
export CUDA_VISIBLE_DEVICES="3"



# python main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 0 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --learning_rate 1.0 --linear_lr 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_05_08

# python main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 1 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --learning_rate 1.0 --linear_lr 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_05_08

# python main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 2 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --learning_rate 1.0 --linear_lr 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_05_08

# python main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 3 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --learning_rate 1.0 --linear_lr 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_05_08

# python main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 4 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --learning_rate 1.0 --linear_lr 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_05_08



python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --cosine --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --linear_lr 0.1 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_05_08

python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --cosine --seed 1 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --linear_lr 0.1 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_05_08

python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --cosine --seed 2 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --linear_lr 0.1 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_05_08

python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --cosine --seed 3 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --linear_lr 0.1 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_05_08

python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --cosine --seed 4 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --linear_lr 0.1 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_05_08






