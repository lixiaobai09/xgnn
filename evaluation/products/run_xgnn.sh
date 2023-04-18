#!/bin/bash 
# comm
MY_DIR="$(dirname $0)"
TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")

dgl_dir=${MY_DIR}/../../example/dgl/multi_gpu/
sgnn_dir=${MY_DIR}/../../example/samgraph/sgnn/

export SAMGRAPH_HUGE_PAGE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

set -x

# config
log_dir=${MY_DIR}/run-logs/${TIME_STAMPS}
num_epoch=10
dataset=products

mkdir -p "$log_dir"

log=${log_dir}/xgnn_gcn_pr
python ${sgnn_dir}/train_gcn.py --num-worker 4 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err

log=${log_dir}/xgnn_gcn_pr_bk
python ${sgnn_dir}/train_gcn.py --num-worker 4 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err
