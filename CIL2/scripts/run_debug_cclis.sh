


export CUDA_VISIBLE_DEVICES="0"


python ./main.py --method cclis --mem_type ring --dataset cifar10 --batch_size 512 --cosine --seed 0 \
                 --temp_cclis 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
                 --learning_rate 1.0 --linear_learning_rate 0.5 --learning_rate_prototypes 0.01 --mem_size 500 --epochs 2 --start_epoch 2 --epoch_save --log_name test2  --date 2025_0902



# python main.py --method cclis --cosine --start_epoch 1 --epochs 1 --val_freq 1 --linear_epochs 3 --log_name test --date 0906