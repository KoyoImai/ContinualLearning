# importance samplingによるバッファに保存するデータをランダムに変更



export CUDA_VISIBLE_DEVICES="0"


# # デバッグ用
python main.py --method cclis-wo-is --mem_type ring --dataset cifar10 --batch_size 512 --seed 0 \
               --temp 0.5 --current_temp 0.2 --past_temp 0.1 --distill_type PRD --distill_power 0.6 \
               --learning_rate 1.0 --learning_rate_prototypes 0.01 --mem_size 2000 --epochs 1 --start_epoch 1 --log_name practice --date 2025_04_24







