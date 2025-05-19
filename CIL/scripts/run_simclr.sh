
export CUDA_VISIBLE_DEVICES="2"



python main.py --method simclr --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --temp 0.07 --mem_size 2000 --seed 0 --log_name simclr --date 2025_05_19

python main.py --method simclr --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --temp 0.07 --mem_size 2000 --seed 1 --log_name simclr --date 2025_05_19

python main.py --method simclr --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --temp 0.07 --mem_size 2000 --seed 2 --log_name simclr --date 2025_05_19

python main.py --method simclr --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --temp 0.07 --mem_size 2000 --seed 3 --log_name simclr --date 2025_05_19

python main.py --method simclr --mem_type ring --dataset cifar10 --batch_size 512 --epochs 100 --start_epoch 500 \
               --temp 0.07 --mem_size 2000 --seed 4 --log_name simclr --date 2025_05_19





python main.py --method simclr --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --temp 0.07 --mem_size 2000 --seed 0 --log_name simclr --date 2025_05_19

python main.py --method simclr --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --temp 0.07 --mem_size 2000 --seed 1 --log_name simclr --date 2025_05_19

python main.py --method simclr --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --temp 0.07 --mem_size 2000 --seed 2 --log_name simclr --date 2025_05_19

python main.py --method simclr --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --temp 0.07 --mem_size 2000 --seed 3 --log_name simclr --date 2025_05_19

python main.py --method simclr --mem_type ring --dataset cifar100 --batch_size 512 --epochs 100 --start_epoch 500 \
               --temp 0.07 --mem_size 2000 --seed 4 --log_name simclr --date 2025_05_19




