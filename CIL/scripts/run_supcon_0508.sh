

export CUDA_VISIBLE_DEVICES="0"



python main.py --method supcon --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --linear_momentum 0.9 --linear_lr 0.5 --temp 0.1 --mem_size 0 --seed 0 --log_name practice --date 2025_05_08 > /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/output0509_cifar10.txt

python main.py --method supcon --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --linear_momentum 0.9 --linear_lr 0.5 --temp 0.1 --mem_size 0 --seed 1 --log_name practice --date 2025_05_08

python main.py --method supcon --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --linear_momentum 0.9 --linear_lr 0.5 --temp 0.1 --mem_size 0 --seed 2 --log_name practice --date 2025_05_08

python main.py --method supcon --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --linear_momentum 0.9 --linear_lr 0.5 --temp 0.1 --mem_size 0 --seed 3 --log_name practice --date 2025_05_08

python main.py --method supcon --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --linear_momentum 0.9 --linear_lr 0.5 --temp 0.1 --mem_size 0 --seed 4 --log_name practice --date 2025_05_08






python main.py --method supcon --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --linear_momentum 0.9 --linear_lr 0.1 --temp 0.1 --mem_size 0 --seed 0 --log_name practice --date 2025_05_08 > /home/kouyou/ContinualLearning/survey/CIL/logs/cclis/output0509_cifar100.txt

python main.py --method supcon --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --linear_momentum 0.9 --linear_lr 0.1 --temp 0.1 --mem_size 0 --seed 1 --log_name practice --date 2025_05_08

python main.py --method supcon --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --linear_momentum 0.9 --linear_lr 0.1 --temp 0.1 --mem_size 0 --seed 2 --log_name practice --date 2025_05_08

python main.py --method supcon --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --linear_momentum 0.9 --linear_lr 0.1 --temp 0.1 --mem_size 0 --seed 3 --log_name practice --date 2025_05_08

python main.py --method supcon --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --learning_rate 1.0 --linear_momentum 0.9 --linear_lr 0.1 --temp 0.1 --mem_size 0 --seed 4 --log_name practice --date 2025_05_08



