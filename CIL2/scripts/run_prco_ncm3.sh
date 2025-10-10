


export CUDA_VISIBLE_DEVICES="3"




# 基本設定（log_name関連）
export METHOD="prco-fimclv2"
export MEM_TYPE="kmeans"
export MEM_SIZE=500
export DATASET="cifar100"
export SEED=0
export LOG_NAME="prco-fimclv2-efm-lambda5-kmeans"
export DATE="2025_1002"






python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --target_task 4 --target_epoch 140


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --target_task 0 --target_epoch 500


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --target_task 1 --target_epoch 140


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --target_task 2 --target_epoch 140


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --target_task 3 --target_epoch 140


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --target_task 4 --target_epoch 140







# for TASKID in {1..5} ; do
#     echo ${TASKID}
#     for EPOCH in {1..100} ; do
#         python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
#                         --target_task ${TASKID} --target_epoch ${EPOCH}
#     done
# done



# for EPOCH in {480..500} ; do
#     python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
#                     --target_task 0 --target_epoch ${EPOCH}
# done

