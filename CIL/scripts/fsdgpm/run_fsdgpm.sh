
export CUDA_VISIBLE_DEVICES="0"

# cifar100
# python main.py --method er --mem_type ring --dataset cifar100 \
#                --batch_size 10 --learning_rate 0.03 --mem_size 0 --epochs 100


# cifar100
# python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 0\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --log_name practice --date 2025_04_12\
#                 --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm
# python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 0\
#                --learning_rate 0.03 --mem_size 2000 --epochs 1 --log_name practice --date 2025_04_12\
#                 --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 10 --second_order --fsdgpm_method xdgpm


# scheduler = lr_scheduler.OneCycleLRを使用
# python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 0\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name practice --date 2025_04_15\
#                 --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm



# # scheduler = lr_scheduler.OneCycleLRを使用
# python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 0\
#                --learning_rate 0.03 --mem_size 2000 --epochs 100 --start_epoch 100 --log_name practice5 --date 2025_04_15\
#                 --sharpness --threshold 0.975 --thres_add 0.003 --mem_batch_size 250 --second_order --fsdgpm_method xdgpm


# scheduler = lr_scheduler.OneCycleLRを使用
# python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 0\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name practice --date 2025_04_16\
#                 --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm

# python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 1\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name practice --date 2025_04_16\
#                 --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm

# python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 2\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name practice --date 2025_04_16\
#                 --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm

# python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 3\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name practice --date 2025_04_16\
#                 --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm

# python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 4\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name practice --date 2025_04_16\
#                 --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm






# fc層は制約を除く
python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 0\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name fs-dgpm --date 2025_04_17\
                --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm

python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 1\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name fs-dgpm --date 2025_04_17\
                --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm

python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 2\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name fs-dgpm --date 2025_04_17\
                --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm

python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 3\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name fs-dgpm --date 2025_04_17\
                --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm

python main.py --method fs-dgpm --mem_type ring --dataset cifar100 --batch_size 64 --seed 4\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name fs-dgpm --date 2025_04_17\
                --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm










