


export CUDA_VISIBLE_DEVICES="0,1"


python main.py --method cclis --cosine --start_epoch 1 --epochs 1 --val_freq 1 --linear_epochs 3 --log_name test --date 0902