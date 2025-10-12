


export CUDA_VISIBLE_DEVICES="1"




# 基本設定（log_name関連）
export METHOD="prco"
export MEM_TYPE="kmeans"
export MEM_SIZE=500
export DATASET="cifar100"
export SEED=0
export LOG_NAME="prco-efm-lambda5-kmeans"
export DATE="2025_1011"

export FEAT_DIM=128

# python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
#                 --target_task 1 --target_epoch 9




python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 4 --target_epoch 100
                

python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 0 --target_epoch 500


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 1 --target_epoch 100


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 2 --target_epoch 100


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 3 --target_epoch 100


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 4 --target_epoch 100







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

