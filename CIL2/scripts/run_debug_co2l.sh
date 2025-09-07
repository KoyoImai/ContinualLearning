

export CUDA_VISIBLE_DEVICES="0"

# python main.py --method co2l --cosine --start_epoch 11 --epochs 11 --val_freq 10 --linear_epochs 3 --log_name test --date 0906

python ./main.py --method co2l --mem_type ring --dataset cifar10 --batch_size 512 --epochs 11 --start_epoch 11  --epoch_save \
               --learning_rate 0.5 --mem_size 500 --current_temp 0.2 --past_temp 0.01 --distill_power 1.0 \
               --seed 0 --linear_learning_rate 1.0 --log_name test2 --date 2025_0906



