

export CUDA_VISIBLE_DEVICES="0"



python ./main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 0 \
                 --temp_cclis 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --uloss_weight 1.0 \
                 --learning_rate 1.0 --linear_learning_rate 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --epoch_save --log_name cclis-w-u1 --date 2025_0902




python ./main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 0 \
                 --temp_cclis 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --uloss_weight 0.5 \
                 --learning_rate 1.0 --linear_learning_rate 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --epoch_save --log_name cclis-w-u05 --date 2025_0902



python ./main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 0 \
                 --temp_cclis 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --uloss_weight 0.1 \
                 --learning_rate 1.0 --linear_learning_rate 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --epoch_save --log_name cclis-w-u01 --date 2025_0902



python ./main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 0 \
                 --temp_cclis 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --uloss_weight 0.1 \
                 --learning_rate 1.0 --linear_learning_rate 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --epoch_save --log_name cclis-w-u0 --date 2025_0902










python ./main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --cosine --seed 0 \
                 --temp_cclis 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --uloss_weight 1.0 \
                 --learning_rate 1.0 --linear_learning_rate 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --epoch_save --log_name cclis-w-u1  --date 2025_0902


python ./main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --cosine --seed 0 \
                 --temp_cclis 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --uloss_weight 0.5 \
                 --learning_rate 1.0 --linear_learning_rate 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --epoch_save --log_name cclis-w-u05  --date 2025_0902


python ./main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --cosine --seed 0 \
                 --temp_cclis 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --uloss_weight 0.1 \
                 --learning_rate 1.0 --linear_learning_rate 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --epoch_save --log_name cclis-w-u01  --date 2025_0902


python ./main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --cosine --seed 0 \
                 --temp_cclis 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 --uloss_weight 0.0 \
                 --learning_rate 1.0 --linear_learning_rate 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 100 --start_epoch 500 --epoch_save --log_name cclis-w-u0  --date 2025_0902









