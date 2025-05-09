
# Samplerによるバッチ作成を行わない



export CUDA_VISIBLE_DEVICES="0"



# cifar10
python main.py --method cclis-wo-ss --mem_type ring --dataset cifar10 --batch_size 512 --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo-ss --date 2025_05_03

python main.py --method cclis-wo-ss --mem_type ring --dataset cifar10 --batch_size 512 --seed 1 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo-ss --date 2025_05_03 > /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-wo-ss/output_cifar10.txt

python main.py --method cclis-wo-ss --mem_type ring --dataset cifar10 --batch_size 512 --seed 2 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo-ss --date 2025_05_03

python main.py --method cclis-wo-ss --mem_type ring --dataset cifar10 --batch_size 512 --seed 3 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo-ss --date 2025_05_03

python main.py --method cclis-wo-ss --mem_type ring --dataset cifar10 --batch_size 512 --seed 4 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo-ss --date 2025_05_03






# cifar100
python main.py --method cclis-wo-ss --mem_type ring --dataset cifar100 --batch_size 512 --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo-ss --date 2025_05_03

python main.py --method cclis-wo-ss --mem_type ring --dataset cifar100 --batch_size 512 --seed 1 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo-ss --date 2025_05_03

python main.py --method cclis-wo-ss --mem_type ring --dataset cifar100 --batch_size 512 --seed 2 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo-ss --date 2025_05_03 > /home/kouyou/ContinualLearning/survey/CIL/logs/cclis-wo-ss/output_cifar100.txt

python main.py --method cclis-wo-ss --mem_type ring --dataset cifar100 --batch_size 512 --seed 3 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo-ss --date 2025_05_03

python main.py --method cclis-wo-ss --mem_type ring --dataset cifar100 --batch_size 512 --seed 4 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 100 --start_epoch 500 --log_name cclis-wo-ss --date 2025_05_03





# # デバッグ用
# python main.py --method cclis-wo-ss --mem_type ring --dataset cifar10 --batch_size 512 --seed 0 \
#                --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
#                --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 1 --start_epoch 1 --log_name practice --date 2025_04_24 




