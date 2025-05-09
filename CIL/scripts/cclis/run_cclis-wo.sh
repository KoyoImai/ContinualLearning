


export CUDA_VISIBLE_DEVICES="1"




# cifar10
python main.py --method cclis-wo --mem_type ring --dataset cifar10 --batch_size 512 --num_workers 4 --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo --date 2025_05_03

python main.py --method cclis-wo --mem_type ring --dataset cifar10 --batch_size 512 --num_workers 4 --seed 1 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo --date 2025_05_03

python main.py --method cclis-wo --mem_type ring --dataset cifar10 --batch_size 512 --num_workers 4 --seed 2 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo --date 2025_05_03 > /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-wo/output_cifar10.txt

python main.py --method cclis-wo --mem_type ring --dataset cifar10 --batch_size 512 --num_workers 4 --seed 3 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo --date 2025_05_03

python main.py --method cclis-wo --mem_type ring --dataset cifar10 --batch_size 512 --num_workers 4 --seed 4 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo --date 2025_05_03







# cifar100
python main.py --method cclis-wo --mem_type ring --dataset cifar100 --batch_size 512 --num_workers 4 --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo --date 2025_05_03

python main.py --method cclis-wo --mem_type ring --dataset cifar100 --batch_size 512 --num_workers 4 --seed 1 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo --date 2025_05_03

python main.py --method cclis-wo --mem_type ring --dataset cifar100 --batch_size 512 --num_workers 4 --seed 2 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo --date 2025_05_03 > /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-wo/output_cifar100.txt

python main.py --method cclis-wo --mem_type ring --dataset cifar100 --batch_size 512 --num_workers 4 --seed 3 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo --date 2025_05_03

python main.py --method cclis-wo --mem_type ring --dataset cifar100 --batch_size 512 --num_workers 4 --seed 4 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo --date 2025_05_03










# # cifar10
# python main.py --method cclis-wo --mem_type ring --dataset cifar10 --batch_size 512 --num_workers 4 --seed 0 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name practice --date 2025_04_24 








