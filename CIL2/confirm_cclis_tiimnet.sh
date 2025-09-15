


export CUDA_VISIBLE_DEVICES="0"


# 基本設定（log_name関連）
export METHOD="cclis"
export MEM_TYPE="ring"
export MEM_SIZE=500
export DATASET="tiny-imagenet"
export SEED=0
export LOG_NAME="cclis"
export DATE="2025_0910"

# 学習のハイパラ関連
export BATCH_SIZE=512
export LEARNING_RATE=1.0
export LEARNING_RATE_PROTOTYPES=0.01

export EPOCHS=50
export START_EPOCH=500


# CCLIS特有設定
export TEMP_CCLIS=0.5
export CURRENT_TEMP=0.2
export PAST_TEMP=0.1
export DISTILL_TYPE="PRD"
export DISTILL_POWER=0.6


# 線形分類関連のハイパラ
export LINEAR_LEARNING_RATE=0.5
export LINEAR_EPOCHS=100
export VAL_FREQ=1000




python main_linear4timnet.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                                                        --linear_learning_rate ${LINEAR_LEARNING_RATE} --linear_epochs ${LINEAR_EPOCHS} \
                                                        --target_task 9 --target_epoch 50

