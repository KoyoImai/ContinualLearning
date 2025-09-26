

export CUDA_VISIBLE_DEVICES="0"




# 基本設定（log_name関連）
export METHOD="co2l"
export MEM_TYPE="ring"
export MEM_SIZE=500
export DATASET="cifar100"
export SEED=0
export LOG_NAME="test2"
export DATE="2025_0907"

# 学習のハイパラ関連
export BATCH_SIZE=512
export LEARNING_RATE=0.5
export EPOCHS=15
export START_EPOCH=15


# Co2L特有設定
export TEMP_CO2L=0.5
export CURRENT_TEMP=0.2
export PAST_TEMP=0.01
export DISTILL_POWER=0.6


# 線形分類関連のハイパラ
export LINEAR_LEARNING_RATE=0.5
export LINEAR_EPOCHS=2
export VAL_FREQ=1000




# python ./main.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
#                  --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --epochs ${EPOCHS} --start_epoch ${START_EPOCH} \
#                  --temp_co2l ${TEMP_CO2L} --current_temp ${CURRENT_TEMP} --past_temp ${PAST_TEMP} --distill_power ${DISTILL_POWER} \
#                  --linear_learning_rate ${LINEAR_LEARNING_RATE} --linear_epochs ${LINEAR_EPOCHS} --val_freq ${VAL_FREQ} \
#                  --epoch_save 


# python ./main_linear.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
#                         --linear_learning_rate ${LINEAR_LEARNING_RATE} --linear_epochs ${LINEAR_EPOCHS} \
#                         --target_task 0 --target_epoch 1



python ./main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                        --linear_learning_rate ${LINEAR_LEARNING_RATE} --linear_epochs ${LINEAR_EPOCHS} \
                        --target_task 0 --target_epoch 14


