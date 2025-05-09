
export CUDA_VISIBLE_DEVICES="1"




# fc層は制約を除く
python main.py --method fs-dgpm --mem_type ring --dataset cifar10 --batch_size 64 --seed 0\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name fs-dgpm --date 2025_04_19\
                --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm

python main.py --method fs-dgpm --mem_type ring --dataset cifar10 --batch_size 64 --seed 1\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name fs-dgpm --date 2025_04_19\
                --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm

python main.py --method fs-dgpm --mem_type ring --dataset cifar10 --batch_size 64 --seed 2\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name fs-dgpm --date 2025_04_19\
                --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm

python main.py --method fs-dgpm --mem_type ring --dataset cifar10 --batch_size 64 --seed 3\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name fs-dgpm --date 2025_04_19\
                --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm

python main.py --method fs-dgpm --mem_type ring --dataset cifar10 --batch_size 64 --seed 4\
               --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name fs-dgpm --date 2025_04_19\
                --sharpness --threshold 0.97 --thres_add 0.003 --mem_batch_size 125 --second_order --fsdgpm_method xdgpm




