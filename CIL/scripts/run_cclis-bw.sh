
export CUDA_VISIBLE_DEVICES="1"



python main.py --method cclis-bw --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --bw_lambd 0.1\
               --learning_rate 1.0 --linear_lr 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name lambd01 --date 2025_05_11

python main.py --method cclis-bw --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --bw_lambd 0.05\
               --learning_rate 1.0 --linear_lr 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name lambd005 --date 2025_05_11
           
python main.py --method cclis-bw --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --bw_lambd 0.01\
               --learning_rate 1.0 --linear_lr 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name lambd001 --date 2025_05_11

python main.py --method cclis-bw --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --bw_lambd 0.005\
               --learning_rate 1.0 --linear_lr 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name lambd0005 --date 2025_05_11 

python main.py --method cclis-bw --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --bw_lambd 0.001\
               --learning_rate 1.0 --linear_lr 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --log_name lambd0001 --date 2025_05_11        






