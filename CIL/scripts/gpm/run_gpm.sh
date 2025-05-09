

export CUDA_VISIBLE_DEVICES="1"


# cifar10
# python main.py --method gpm --mem_type ring --dataset cifar10 \
#                --batch_size 64 --learning_rate 0.01 --mem_size 0 --epochs 100



# cifar100（論文と補足資料参考）
# python main.py --method gpm --mem_type ring --dataset cifar100 --batch_size 64  --seed 0\
#                --learning_rate 0.01 --mem_size 0 --epochs 100 --log_name gpm --date 2025_04_09

# python main.py --method gpm --mem_type ring --dataset cifar100 --batch_size 64  --seed 1\
#                --learning_rate 0.01 --mem_size 0 --epochs 100 --log_name gpm --date 2025_04_09

# python main.py --method gpm --mem_type ring --dataset cifar100 --batch_size 64  --seed 2\
#                --learning_rate 0.01 --mem_size 0 --epochs 100 --log_name gpm --date 2025_04_09

# python main.py --method gpm --mem_type ring --dataset cifar100 --batch_size 64  --seed 3\
#                --learning_rate 0.01 --mem_size 0 --epochs 100 --log_name gpm --date 2025_04_09

# python main.py --method gpm --mem_type ring --dataset cifar100 --batch_size 64  --seed 4\
#                --learning_rate 0.01 --mem_size 0 --epochs 100 --log_name gpm --date 2025_04_09


python main.py --method gpm --mem_type ring --dataset cifar100 --batch_size 64  --seed 0\
               --learning_rate 0.01 --mem_size 2000 --epochs 100 --log_name gpm --date 2025_04_09

python main.py --method gpm --mem_type ring --dataset cifar100 --batch_size 64  --seed 1\
               --learning_rate 0.01 --mem_size 2000 --epochs 100 --log_name gpm --date 2025_04_09

python main.py --method gpm --mem_type ring --dataset cifar100 --batch_size 64  --seed 2\
               --learning_rate 0.01 --mem_size 2000 --epochs 100 --log_name gpm --date 2025_04_09

python main.py --method gpm --mem_type ring --dataset cifar100 --batch_size 64  --seed 3\
               --learning_rate 0.01 --mem_size 2000 --epochs 100 --log_name gpm --date 2025_04_09

python main.py --method gpm --mem_type ring --dataset cifar100 --batch_size 64  --seed 4\
               --learning_rate 0.01 --mem_size 2000 --epochs 100 --log_name gpm --date 2025_04_09





# python main.py --method gpm --mem_type ring --dataset cifar100 --batch_size 64 --learning_rate 0.01 --mem_size 0 --epochs 100
# python main.py --method gpm --mem_type ring --dataset cifar100 --seed 0 --batch_size 64 \
#                --learning_rate 0.01 --mem_size 0 --epochs 100 --log_name gpm --date 2025_04_09


# # tiny-imagenet
# python main.py --method gpm --mem_type ring --dataset tiny-imagenet \
#                --batch_size 64 --learning_rate 0.01 --mem_size 0 --epochs 100