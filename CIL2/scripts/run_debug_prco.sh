


export CUDA_VISIBLE_DEVICES="3"



# 基本設定（log_name関連）
export METHOD="prco"
export MEM_TYPE="ring"
export MEM_SIZE=500
export DATASET="cifar10"
export SEED=0
export LOG_NAME="test-normal-distill"
export DATE="2025_0910"


# 学習のハイパラ関連
export BATCH_SIZE=512
export LEARNING_RATE=1.0
export LEARNING_RATE_PROTOTYPES=0.01

export EPOCHS=50
export START_EPOCH=50


# PRCO特有の設定
export TEMP_PRCO=0.5
export CURRENT_TEMP=0.2
export PAST_TEMP=0.1
export DISTILL_TYPE="ND"    # PRD. EFC, ND
export DISTILL_POWER=1.0


# 線形分類関連のハイパラ
export LINEAR_LEARNING_RATE=0.5
export LINEAR_EPOCHS=100
export VAL_FREQ=1000





# python main.py --method prco --dataset cifar10 --distill_type None --cosine --start_epoch 11 --epochs 11 --val_freq 10 --linear_epochs 3 --log_name test --date 0906


python main.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --batch_size ${BATCH_SIZE} --learning_rate ${LEARNING_RATE} --learning_rate_prototypes ${LEARNING_RATE_PROTOTYPES} \
                --epochs ${EPOCHS} --start_epoch ${START_EPOCH} \
                --temp_prco ${TEMP_PRCO} --current_temp ${CURRENT_TEMP} --past_temp ${PAST_TEMP} --distill_type ${DISTILL_TYPE} --distill_power ${DISTILL_POWER} \
                --linear_learning_rate ${LINEAR_LEARNING_RATE} --linear_epochs ${LINEAR_EPOCHS} --val_freq ${VAL_FREQ} \
                --epoch_save --cosine 



# for TASKID in {2..10} ; do
#     echo ${TASKID}
#     for EPOCH in {1..3} ; do
#         python ./main_linear.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
#                 --linear_learning_rate ${LINEAR_LEARNING_RATE} --linear_epochs ${LINEAR_EPOCHS} \
#                 --target_task ${TASKID} --target_epoch ${EPOCH}
#     done
# done