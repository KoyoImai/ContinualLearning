
export CUDA_VISIBLE_DEVICES="2"

# # cifar100
# python main.py --method er --mem_type ring --dataset tiny-imagenet --batch_size 10 --seed 0\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name erring --date 2025_04_20

# # cifar100
# python main.py --method er --mem_type ring --dataset tiny-imagenet --batch_size 10 --seed 1\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name erring --date 2025_04_20

# # cifar100
# python main.py --method er --mem_type ring --dataset tiny-imagenet --batch_size 10 --seed 2\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name erring --date 2025_04_20

# # cifar100
# python main.py --method er --mem_type ring --dataset tiny-imagenet --batch_size 10 --seed 3\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name erring --date 2025_04_20

# # cifar100
# python main.py --method er --mem_type ring --dataset tiny-imagenet --batch_size 10 --seed 4\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name erring --date 2025_04_20





# python main.py --method er --mem_type reservoir --dataset tiny-imagenet --batch_size 10 --seed 0\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name er --date 2025_04_20

# # cifar100
# python main.py --method er --mem_type reservoir --dataset tiny-imagenet --batch_size 10 --seed 1\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name er --date 2025_04_20

# cifar100
python main.py --method er --mem_type reservoir --dataset tiny-imagenet --batch_size 10 --seed 2\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name er --date 2025_04_20

# cifar100
python main.py --method er --mem_type reservoir --dataset tiny-imagenet --batch_size 10 --seed 3\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name er --date 2025_04_20

# cifar100
python main.py --method er --mem_type reservoir --dataset ctiny-imagenet --batch_size 10 --seed 4\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name er --date 2025_04_20






python main.py --method er --mem_type reservoir --dataset tiny-imagenet --batch_size 10 --seed 0\
               --learning_rate 0.03 --mem_size 0 --epochs 50 --start_epoch 100 --log_name fintune --date 2025_04_20

# cifar100
python main.py --method er --mem_type reservoir --dataset tiny-imagenet --batch_size 10 --seed 1\
               --learning_rate 0.03 --mem_size 0 --epochs 50 --start_epoch 100 --log_name fintune --date 2025_04_20

# cifar100
python main.py --method er --mem_type reservoir --dataset tiny-imagenet --batch_size 10 --seed 2\
               --learning_rate 0.03 --mem_size 0 --epochs 50 --start_epoch 100 --log_name fintune --date 2025_04_20

# cifar100
python main.py --method er --mem_type reservoir --dataset tiny-imagenet --batch_size 10 --seed 3\
               --learning_rate 0.03 --mem_size 0 --epochs 50 --start_epoch 100 --log_name fintune --date 2025_04_20

# cifar100
python main.py --method er --mem_type reservoir --dataset tiny-imagenet --batch_size 10 --seed 4\
               --learning_rate 0.03 --mem_size 0 --epochs 50 --start_epoch 100 --log_name fintune --date 2025_04_20
