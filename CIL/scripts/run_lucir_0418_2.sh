
export CUDA_VISIBLE_DEVICES="1"


python main.py --method lucir --mem_type herding --dataset cifar10 --batch_size 128 --seed 0\
                 --learning_rate 0.1 --mem_size 2000 --epochs 160 --start_epoch 160 --log_name lucir --date 2025_04_18

python main.py --method lucir --mem_type herding --dataset cifar10 --batch_size 128 --seed 1\
                 --learning_rate 0.1 --mem_size 2000 --epochs 160 --start_epoch 160 --log_name lucir --date 2025_04_18

python main.py --method lucir --mem_type herding --dataset cifar10 --batch_size 128 --seed 2\
                 --learning_rate 0.1 --mem_size 2000 --epochs 160 --start_epoch 160 --log_name lucir --date 2025_04_18

python main.py --method lucir --mem_type herding --dataset cifar10 --batch_size 128 --seed 3\
                 --learning_rate 0.1 --mem_size 2000 --epochs 160 --start_epoch 160 --log_name lucir --date 2025_04_18

python main.py --method lucir --mem_type herding --dataset cifar10 --batch_size 128 --seed 4\
                 --learning_rate 0.1 --mem_size 2000 --epochs 160 --start_epoch 160 --log_name lucir --date 2025_04_18