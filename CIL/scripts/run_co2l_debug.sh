
export CUDA_VISIBLE_DEVICES="2"





python main.py --method co2l --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 15 \
               --learning_rate 0.5 --mem_size 500 --current_temp 0.2 --past_temp 0.01 --distill_power 1.0 \
               --grad_analysis --grad_analysis_freq 5 --seed 0 --log_name debug --date 2025_0804

