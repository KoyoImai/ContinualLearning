


export CUDA_VISIBLE_DEVICES="0"

# 基本設定
export METHOD="cclis"
export MEM_TYPE="ring"
export DATASET="cifar10"
export BATCH_SIZE=512
export SEED=0

export LEARNING_RATE=1.0
export LEARNING_RATE_PROTOTYPES=0.01

export LINEAR_LEARNING_RATE=0.5
export LINEAR_EPOCHS=2

export MEM_SIZE=500
export EPOCHS=2
export START_EPOCH=2

export LOG_NAME="test2"
export DATE="2025_0902"





# CCLIS特有設定
export TEMP_CCLIS=0.5
export CURRENT_TEMP=02
export PAST_TEMP=0.1
export DISTILL_TYPE="PRD"
export DISTILL_POWER=0.6





# python ./main.py --method ${METHOD} --mem_type ${MEM_TYPE} --dataset ${DATASET} --batch_size ${BATCH_SIZE} --cosine --seed ${SEED} \
#                  --temp_cclis ${TEMP_CCLIS} --current_temp ${CURRENT_TEMP} --past_temp ${PAST_TEMP} --distill_type ${DISTILL_TYPE} --distill_power ${DISTILL_POWER} \
#                  --learning_rate ${LEARNING_RATE} --linear_learning_rate ${LINEAR_LEARNING_RATE} --learning_rate_prototypes ${LEARNING_RATE_PROTOTYPES} \
#                  --mem_size ${MEM_SIZE} --epochs ${EPOCHS} --start_epoch ${START_EPOCH} --epoch_save \
#                  --log_name ${LOG_NAME}  --date ${DATE}


python ./main_linear.py --method ${METHOD} --mem_type ${MEM_TYPE} --dataset ${DATASET} --cosine --seed ${SEED} \
                        --temp_cclis ${TEMP_CCLIS} --current_temp ${CURRENT_TEMP} --past_temp ${PAST_TEMP} --distill_type ${DISTILL_TYPE} --distill_power ${DISTILL_POWER} \
                        --learning_rate ${LEARNING_RATE} --learning_rate_prototypes ${LEARNING_RATE_PROTOTYPES} \
                        --linear_learning_rate ${LINEAR_LEARNING_RATE} --linear_epochs ${LINEAR_EPOCHS} \
                        --epoch_save --log_name ${LOG_NAME}  --date ${DATE} --target_task 1 --target_epoch 1


# python main.py --method cclis --cosine --start_epoch 1 --epochs 1 --val_freq 1 --linear_epochs 3 --log_name test --date 0906

















# 基本設定（log_name関連）
export METHOD="cclis"
export MEM_TYPE="ring"
export MEM_SIZE=500
export DATASET="cifar10"
export SEED=0
export LOG_NAME="cclis-v3"
export DATE="2025_0907"

# 学習のハイパラ関連
export BATCH_SIZE=512
export LEARNING_RATE=1.0
export LEARNING_RATE_PROTOTYPES=0.01

export EPOCHS=100
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





# python ./ContinualLearning3/CIL2/main.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
#                 --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --learning_rate_prototypes ${LEARNING_RATE_PROTOTYPES} \
#                 --epochs ${EPOCHS} --start_epoch ${START_EPOCH} \
#                 --temp_cclis ${TEMP_CCLIS} --current_temp ${CURRENT_TEMP} --past_temp ${PAST_TEMP} --distill_type ${DISTILL_TYPE} --distill_power ${DISTILL_POWER} \ 
#                 --linear_learning_rate ${LINEAR_LEARNING_RATE} --linear_epochs ${LINEAR_EPOCHS} --val_freq ${VAL_FREQ} \
#                 --epoch_save --cosine 



# 線形分類の実行例
# python ./ContinualLearning3/CIL2/main_linear.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
#                 --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --learning_rate_prototypes ${LEARNING_RATE_PROTOTYPES} \
#                 --epochs ${EPOCHS} --start_epoch ${START_EPOCH} \
#                 --temp_cclis ${TEMP_CCLIS} --current_temp ${CURRENT_TEMP} --past_temp ${PAST_TEMP} --distill_type ${DISTILL_TYPE} --distill_power ${DISTILL_POWER} \ 
#                 --linear_learning_rate ${LINEAR_LEARNING_RATE} --linear_epochs ${LINEAR_EPOCHS} --val_freq ${VAL_FREQ} \
#                 --epoch_save --cosine 



for TASKID in {2..5} ; do
    echo ${TASKID}
    for EPOCH in {1..100} ; do
        python ./ContinualLearning3/CIL2/main_linear.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                        --linear_learning_rate ${LINEAR_LEARNING_RATE} --linear_epochs ${LINEAR_EPOCHS} \
                        --target_task ${TASKID} --target_epoch ${EPOCH}
    done
done


for EPOCH in {450..500} ; do
    python ./ContinualLearning3/CIL2/main_linear.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                    --linear_learning_rate ${LINEAR_LEARNING_RATE} --linear_epochs ${LINEAR_EPOCHS} \
                    --target_task 0 --target_epoch ${EPOCH}
done
