#!/bin/bash 
# comm
MY_DIR="$(dirname $0)"
TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")

dgl_dir=${MY_DIR}/../../example/dgl/multi_gpu/
sgnn_dir=${MY_DIR}/../../example/samgraph/sgnn/

export SAMGRAPH_HUGE_PAGE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# config
log_dir=${MY_DIR}/run-logs/${TIME_STAMPS}
dataset=papers100M
ds_short=pa
model=gcn
num_epoch=3

dgl_log=${log_dir}/dgl_${model}_${ds_short}
sgnn_log=${log_dir}/sgnn_${model}_${ds_short}
xgnn_log=${log_dir}/xgnn_${model}_${ds_short}

mkdir -p "$log_dir"

### 1GPU ###
log=${dgl_log}_1wk
python ${dgl_dir}/train_${model}.py --devices 0 --num-epoch ${num_epoch} --dataset ${dataset} \
    --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_1wk
python ${sgnn_dir}/train_${model}.py --num-worker 1 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage 0.07 > ${log}.log 2> ${log}.err

log=${xgnn_log}_1wk
python ${sgnn_dir}/train_${model}.py --num-worker 1 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage 0.07 > ${log}.log 2> ${log}.err

### 2GPU ###
log=${dgl_log}_2wk
python ${dgl_dir}/train_${model}.py --devices 0 1 --num-epoch ${num_epoch} --dataset ${dataset} \
    --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_2wk
python ${sgnn_dir}/train_${model}.py --num-worker 2 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage 0.07 > ${log}.log 2> ${log}.err

log=${xgnn_log}_2wk
python ${sgnn_dir}/train_${model}.py --num-worker 2 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage 0.13 > ${log}.log 2> ${log}.err

# ### 3GPU ###
log=${dgl_log}_3wk
python ${dgl_dir}/train_${model}.py --devices 0 1 2 --num-epoch ${num_epoch} --dataset ${dataset} \
    --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_3wk
python ${sgnn_dir}/train_${model}.py --num-worker 3 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage 0.07 > ${log}.log 2> ${log}.err

log=${xgnn_log}_3wk
python ${sgnn_dir}/train_${model}.py --num-worker 3 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage 0.15 > ${log}.log 2> ${log}.err

# ### 4GPU ###
log=${dgl_log}_4wk
python ${dgl_dir}/train_${model}.py --devices 0 1 2 3 --num-epoch ${num_epoch} --dataset ${dataset} \
    --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_4wk
python ${sgnn_dir}/train_${model}.py --num-worker 4 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage 0.06 > ${log}.log 2> ${log}.err

log=${xgnn_log}_4wk
python ${sgnn_dir}/train_${model}.py --num-worker 4 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage 0.15 > ${log}.log 2> ${log}.err

### 5GPU ###
log=${dgl_log}_5wk
python ${dgl_dir}/train_${model}.py --devices 0 1 2 3 4 --num-epoch ${num_epoch} --dataset ${dataset} \
    --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_5wk
python ${sgnn_dir}/train_${model}.py --num-worker 5 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage 0.07 > ${log}.log 2> ${log}.err

log=${xgnn_log}_5wk
python ${sgnn_dir}/train_${model}.py --num-worker 5 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage 0.14 > ${log}.log 2> ${log}.err

# # 6GPU
log=${dgl_log}_6wk
python ${dgl_dir}/train_${model}.py --devices 0 1 2 3 4 5 --num-epoch ${num_epoch} --dataset ${dataset} \
    --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_6wk
python ${sgnn_dir}/train_${model}.py --num-worker 6 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage 0.07 > ${log}.log 2> ${log}.err

log=${xgnn_log}_6wk
python ${sgnn_dir}/train_${model}.py --num-worker 6 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage 0.15 > ${log}.log 2> ${log}.err

# # 7GPU
log=${dgl_log}_7wk
python ${dgl_dir}/train_${model}.py --devices 0 1 2 3 4 5 6 --num-epoch ${num_epoch} --dataset ${dataset} \
    --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_7wk
python ${sgnn_dir}/train_${model}.py --num-worker 7 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage 0.07 > ${log}.log 2> ${log}.err

log=${xgnn_log}_7wk
python ${sgnn_dir}/train_${model}.py --num-worker 7 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage 0.16 > ${log}.log 2> ${log}.err

# # 8GPU
log=${dgl_log}_8wk
python ${dgl_dir}/train_${model}.py --devices 0 1 2 3 4 5 6 7 --num-epoch ${num_epoch} --dataset ${dataset} \
    --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_8wk
python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage 0.07 > ${log}.log 2> ${log}.err

log=${xgnn_log}_8wk
python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage 0.16 > ${log}.log 2> ${log}.err