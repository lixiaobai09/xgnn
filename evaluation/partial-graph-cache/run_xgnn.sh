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
model=gcn
num_epoch=3
dataset=com-friendster

mkdir -p "$log_dir"

xgnn_cf_gcn_feat_cache_pct=( 0.58 0.54  0.5 0.46 0.42 0.38 0.34 0.30 0.26 0.22 0.18)
xgnn_cf_gcn_graph_cache_pct=(0.0  0.1   0.2 0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0)


xgnn_log=${log_dir}/xgnn_${model}_cf


# all graph in cpu
log=${xgnn_log}_g00
    python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
        --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
        --gpu-extract \
        --unified-memory --unified-memory-percentage 0 1 \
        --cache-percentage ${xgnn_cf_gcn_feat_cache_pct[$i]} \
        > ${log}.log 2> ${log}.err

log=${xgnn_log}_g00-bk
    python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
        --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
        --gpu-extract \
        --unified-memory --unified-memory-percentage 0 1 \
        --cache-percentage ${xgnn_cf_gcn_feat_cache_pct[$i]} \
        > ${log}.log 2> ${log}.err

# partial graph in gpu
for i in `seq 1 10`
do
    log=${xgnn_log}_g${i}0
    python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
        --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
        --gpu-extract \
        --use-dist-graph ${xgnn_cf_gcn_graph_cache_pct[$i]} \
        --cache-percentage ${xgnn_cf_gcn_feat_cache_pct[$i]} \
        > ${log}.log 2> ${log}.err

    log=${xgnn_log}_g${i}0-bk
    python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
        --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
        --gpu-extract \
        --use-dist-graph ${xgnn_cf_gcn_graph_cache_pct[$i]} \
        --cache-percentage ${xgnn_cf_gcn_feat_cache_pct[$i]} \
        > ${log}.log 2> ${log}.err
done