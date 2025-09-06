

export CUDA_VISIBLE_DEVICES="2"




python ./main.py --method co2l --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500  --epoch_save \
               --learning_rate 0.5 --mem_size 500 --current_temp 0.2 --past_temp 0.01 --distill_power 1.0 --uloss_weight 1.0\
               --seed 0 --linear_learning_rate 1.0 --log_name co2l --date 2025_0831




python ./main.py --method co2l --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500  --epoch_save \
               --learning_rate 0.5 --mem_size 500 --current_temp 0.2 --past_temp 0.01 --distill_power 1.0 --uloss_weight 0.0\
               --seed 0 --linear_learning_rate 1.0 --log_name co2l --date 2025_0901




