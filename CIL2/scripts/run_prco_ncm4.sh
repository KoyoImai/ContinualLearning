


export CUDA_VISIBLE_DEVICES="0"




# 基本設定（log_name関連）
export METHOD="prco"
export MEM_TYPE="kmeans"
export MEM_SIZE=500
export DATASET="tiny-imagenet"
export SEED=0
export LOG_NAME="prco-efm-lambda5-kmeans-fdim256"
export DATE="2025_0919"



export FEAT_DIM=256



python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 9 --target_epoch 50


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 0 --target_epoch 500


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 1 --target_epoch 50


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 2 --target_epoch 50


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 3 --target_epoch 50


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 4 --target_epoch 50


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 5 --target_epoch 50


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 6 --target_epoch 50


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 7 --target_epoch 50


python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 8 --target_epoch 50

python main_ncm.py --method ${METHOD} --mem_type ${MEM_TYPE} --mem_size ${MEM_SIZE} --dataset ${DATASET} --seed ${SEED} --log_name ${LOG_NAME} --date ${DATE} \
                --feat_dim ${FEAT_DIM} --target_task 9 --target_epoch 50





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

