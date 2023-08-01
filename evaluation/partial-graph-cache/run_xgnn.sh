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
model=graphsage
num_epoch=3
# dataset=com-friendster

mkdir -p "$log_dir"

# xgnn_cf_gcn_feat_cache_pct=( 0.58 0.54  0.5 0.46 0.42 0.38 0.34 0.30 0.26 0.22 0.18)
# xgnn_cf_gcn_graph_cache_pct=(0.0  0.1   0.2 0.3  0.4  0.5  0.6  0.7  0.8  0.9  1.0)

xgnn_graphsage_graph_cache_pct=(0.0 0.25 0.50 0.75 1.0)
declare -A xgnn_graphsage_feat_cache_pct=(
    ["tw"]="0 0 0 0 0"
    ["pa"]="0 0 0 0 0"
    ["uk"]="0 0 0 0 0"
    ["cf"]="0 0 0 0 0"
)

# input: dataset ds_short
partial_graph() {

dataset=$1
ds_short=$2

xgnn_log=${log_dir}/xgnn_${model}_${ds_short}

local feat_cache_pct=(${xgnn_graphsage_feat_cache_pct[${ds_short}]})

# all graph in cpu
log=${xgnn_log}_g00
python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract \
    --unified-memory --unified-memory-percentage 0 1 \
    --cache-percentage ${feat_cache_pct[0]} \
    > ${log}.log 2> ${log}.err

# log=${xgnn_log}_g00-bk
# python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
#     --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
#     --gpu-extract \
#     --unified-memory --unified-memory-percentage 0 1 \
#     --cache-percentage ${feat_cache_pct[0]} \
#     > ${log}.log 2> ${log}.err

# partial graph in gpu
for i in `seq 1 4`
do
    gc=$((i*25))
    log=${xgnn_log}_g${gc}
    python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
        --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
        --gpu-extract \
        --use-dist-graph ${xgnn_graphsage_graph_cache_pct[$i]} \
        --cache-percentage ${feat_cache_pct[$i]} \
        > ${log}.log 2> ${log}.err

    # log=${xgnn_log}_g${gc}-bk
    # python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    #     --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
    #     --gpu-extract \
    #     --use-dist-graph ${xgnn_graphsage_graph_cache_pct[$i]} \
    #     --cache-percentage ${feat_cache_pct[$i]} \
    #     > ${log}.log 2> ${log}.err
done

}

partial_graph "twitter"    "tw"
partial_graph "papers100M" "pa"
partial_graph "uk-2006-05" "uk"
partial_graph "com-friendster" "cf"
