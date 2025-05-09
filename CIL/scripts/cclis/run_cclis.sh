


export CUDA_VISIBLE_DEVICES="2"



# cifar10
python main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name practice --date 2025_04_24 > ./cclis_output.txt



# python main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --seed 0 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_04_20

# python main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --seed 1 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_04_20

# python main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --seed 2 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_04_20

# python main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --seed 3 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_04_20

# python main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --seed 3 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis --date 2025_04_20


