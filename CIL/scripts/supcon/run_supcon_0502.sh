export CUDA_VISIBLE_DEVICES="1"



python main.py --method supcon-joint --mem_type ring --dataset cifar10 --batch_size 512 --epochs 500 --start_epoch 500 \
               --temp 0.1 --mem_size 0 --seed 0 --log_name practice --date 2025_05_02

python main.py --method supcon-joint --mem_type ring --dataset cifar10 --batch_size 512 --epochs 500 --start_epoch 500 \
               --temp 0.1 --mem_size 0 --seed 1 --log_name practice --date 2025_05_02

python main.py --method supcon-joint --mem_type ring --dataset cifar10 --batch_size 512 --epochs 500 --start_epoch 500 \
               --temp 0.1 --mem_size 0 --seed 2 --log_name practice --date 2025_05_02

python main.py --method supcon-joint --mem_type ring --dataset cifar10 --batch_size 512 --epochs 500 --start_epoch 500 \
               --temp 0.1 --mem_size 0 --seed 3 --log_name practice --date 2025_05_02

python main.py --method supcon-joint --mem_type ring --dataset cifar10 --batch_size 512 --epochs 500 --start_epoch 500 \
               --temp 0.1 --mem_size 0 --seed 4 --log_name practice --date 2025_05_02






python main.py --method supcon-joint --mem_type ring --dataset cifar100 --batch_size 512 --epochs 500 --start_epoch 500 \
               --temp 0.1 --mem_size 0 --seed 0 --log_name practice --date 2025_05_02

python main.py --method supcon-joint --mem_type ring --dataset cifar100 --batch_size 512 --epochs 500 --start_epoch 500 \
               --temp 0.1 --mem_size 0 --seed 1 --log_name practice --date 2025_05_02

python main.py --method supcon-joint --mem_type ring --dataset cifar100 --batch_size 512 --epochs 500 --start_epoch 500 \
               --temp 0.1 --mem_size 0 --seed 2 --log_name practice --date 2025_05_02

python main.py --method supcon-joint --mem_type ring --dataset cifar100 --batch_size 512 --epochs 500 --start_epoch 500 \
               --temp 0.1 --mem_size 0 --seed 3 --log_name practice --date 2025_05_02

python main.py --method supcon-joint --mem_type ring --dataset cifar100 --batch_size 512 --epochs 500 --start_epoch 500 \
               --temp 0.1 --mem_size 0 --seed 4 --log_name practice --date 2025_05_02



