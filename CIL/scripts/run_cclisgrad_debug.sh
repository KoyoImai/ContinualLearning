

export CUDA_VISIBLE_DEVICES="1"



python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --cosine --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --grad_analysis --grad_analysis_freq 1 --wo_is \
               --learning_rate 1.0 --linear_lr 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 10 --start_epoch 2 --epoch_save --log_name distill  --date 2025_0810


# python main.py --method cclis --mem_type ring --dataset cifar100 --batch_size 512 --cosine --seed 0 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --grad_analysis --grad_analysis_freq 1 --wo_is --data_order sparse2coarse \
#                --learning_rate 1.0 --linear_lr 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 10 --start_epoch 2 --epoch_save --log_name distill  --date 2025_0810


