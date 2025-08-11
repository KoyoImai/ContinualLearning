
export CUDA_VISIBLE_DEVICES="3"

# cifar10
# python main.py --method er --mem_type ring --dataset cifar10 --batch_size 100 --seed 0 \
#                --grad_analysis --grad_analysis_freq 1 --print_freq 100  --learning_rate 0.03 --mem_size 500 --epochs 2 --start_epoch 2 --log_name debug --date 2025_0811


python main.py --method er --mem_type ring --dataset cifar10 --batch_size 100 --seed 0 \
               --grad_analysis --grad_analysis_freq 1 --print_freq 100  --learning_rate 0.03 --mem_size 500 --epochs 2 --start_epoch 1 --log_name test --date 2025_0811




# # cifar10
# python main.py --method er --mem_type ring --dataset cifar10 --batch_size 10 --seed 1\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name erring --date 2025_04_19

# # cifar10
# python main.py --method er --mem_type ring --dataset cifar10 --batch_size 10 --seed 2\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name erring --date 2025_04_19

# # cifar10
# python main.py --method er --mem_type ring --dataset cifar10 --batch_size 10 --seed 3\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name erring --date 2025_04_19

# # cifar10
# python main.py --method er --mem_type ring --dataset cifar10 --batch_size 10 --seed 4\
#                --learning_rate 0.03 --mem_size 2000 --epochs 50 --start_epoch 100 --log_name erring --date 2025_04_19