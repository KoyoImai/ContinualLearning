

export CUDA_VISIBLE_DEVICES="3"



python main.py --method supcon --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --mem_size 0 --temp 0.1 --mem_size 0 --seed 0 --log_name offline --date 2025_05_13

python main.py --method supcon --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --mem_size 0 --temp 0.1 --mem_size 0 --seed 1 --log_name offline --date 2025_05_13

python main.py --method supcon --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --mem_size 0 --temp 0.1 --mem_size 0 --seed 2 --log_name offline --date 2025_05_13

python main.py --method supcon --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --mem_size 0 --temp 0.1 --mem_size 0 --seed 3 --log_name offline --date 2025_05_13

python main.py --method supcon --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --mem_size 0 --temp 0.1 --mem_size 0 --seed 4 --log_name offline --date 2025_05_13






python main.py --method supcon --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --mem_size 0 --temp 0.1 --mem_size 0 --seed 0 --log_name offline --date 2025_05_13

python main.py --method supcon --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --mem_size 0 --temp 0.1 --mem_size 0 --seed 1 --log_name offline --date 2025_05_13

python main.py --method supcon --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --mem_size 0 --temp 0.1 --mem_size 0 --seed 2 --log_name offline --date 2025_05_13

python main.py --method supcon --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --mem_size 0 --temp 0.1 --mem_size 0 --seed 3 --log_name offline --date 2025_05_13

python main.py --method supcon --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --mem_size 0 --temp 0.1 --mem_size 0 --seed 4 --log_name offline --date 2025_05_13