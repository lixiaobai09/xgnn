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
num_epoch=3

mkdir -p "$log_dir"

xgnn_pa_gcn_cache_pct=(-1 0.07 0.13 0.15 0.15 0.14 0.15 0.16 0.16)
xgnn_uk_graphsage_cache_pct=(-1 0.01 0.09 0.11 0.12 0.07 0.11 0.10 0.12)

sgnn_pa_gcn_cache_pct=(-1 0.07 0.07 0.07 0.06 0.07 0.07 0.07 0.07)
sgnn_uk_graph_cache_pct=(-1 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01)

declare -A xgnn_cache_percent=(
    ["pa_gcn"]=${xgnn_pa_gcn_cache_pct[@]}
    ["uk_graphsage"]=${xgnn_uk_graphsage_cache_pct[@]}
)

declare -A sgnn_cache_percent=(
    ["pa_gcn"]="${sgnn_pa_gcn_cache_pct[@]}"
    ["uk_graphsage"]="${sgnn_uk_graph_cache_pct[@]}"
)

#input: dataset, ds_short, model
scalability() {

dataset=$1
ds_short=$2
model=$3

xgnn_cache_pct=(${xgnn_cache_percent[${ds_short}_${model}]})
sgnn_cache_pct=(${sgnn_cache_percent[${ds_short}_${model}]})

dgl_log=${log_dir}/dgl_${model}_${ds_short}
sgnn_log=${log_dir}/sgnn_${model}_${ds_short}
xgnn_log=${log_dir}/xgnn_${model}_${ds_short}

dgl_data_root="/graph-learning/samgraph"
if [ "$dataset" = "uk-2006-05" ] || [ "$dataset" = "com-friendster" ]; then
    dgl_data_root="/data/samgraph"
fi

### 1GPU ###
# log=${dgl_log}_1wk
# python ${dgl_dir}/train_${model}.py --devices 0 --num-epoch ${num_epoch} \
#     --dataset ${dataset} --root-path ${dgl_data_root} \
#     --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_1wk
python ${sgnn_dir}/train_${model}.py --num-worker 1 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage ${sgnn_cache_pct[1]} > ${log}.log 2> ${log}.err

log=${xgnn_log}_1wk
python ${sgnn_dir}/train_${model}.py --num-worker 1 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage ${xgnn_cache_pct[1]} > ${log}.log 2> ${log}.err

### 2GPU ###
# log=${dgl_log}_2wk
# python ${dgl_dir}/train_${model}.py --devices 0 1 --num-epoch ${num_epoch} \
#     --dataset ${dataset} --root-path ${dgl_data_root} \
#     --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_2wk
python ${sgnn_dir}/train_${model}.py --num-worker 2 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage ${sgnn_cache_pct[2]} > ${log}.log 2> ${log}.err

log=${xgnn_log}_2wk
python ${sgnn_dir}/train_${model}.py --num-worker 2 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage ${xgnn_cache_pct[2]} > ${log}.log 2> ${log}.err

# ### 3GPU ###
# log=${dgl_log}_3wk
# python ${dgl_dir}/train_${model}.py --devices 0 1 2 --num-epoch ${num_epoch} \
#     --dataset ${dataset} --root-path ${dgl_data_root} \
#     --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_3wk
python ${sgnn_dir}/train_${model}.py --num-worker 3 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage ${sgnn_cache_pct[3]} > ${log}.log 2> ${log}.err

log=${xgnn_log}_3wk
python ${sgnn_dir}/train_${model}.py --num-worker 3 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage ${xgnn_cache_pct[3]} > ${log}.log 2> ${log}.err

# ### 4GPU ###
# log=${dgl_log}_4wk
# python ${dgl_dir}/train_${model}.py --devices 0 1 2 3 --num-epoch ${num_epoch} \
#     --dataset ${dataset} --root-path ${dgl_data_root} \
#     --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_4wk
python ${sgnn_dir}/train_${model}.py --num-worker 4 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage ${sgnn_cache_pct[4]} > ${log}.log 2> ${log}.err

log=${xgnn_log}_4wk
python ${sgnn_dir}/train_${model}.py --num-worker 4 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage ${xgnn_cache_pct[4]} > ${log}.log 2> ${log}.err

### 5GPU ###
# log=${dgl_log}_5wk
# python ${dgl_dir}/train_${model}.py --devices 0 1 2 3 4 --num-epoch ${num_epoch} \
#     --dataset ${dataset} --root-path ${dgl_data_root} \
#     --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_5wk
python ${sgnn_dir}/train_${model}.py --num-worker 5 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage ${sgnn_cache_pct[5]} > ${log}.log 2> ${log}.err

log=${xgnn_log}_5wk
python ${sgnn_dir}/train_${model}.py --num-worker 5 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage ${xgnn_cache_pct[5]} > ${log}.log 2> ${log}.err

# # 6GPU
# log=${dgl_log}_6wk
# python ${dgl_dir}/train_${model}.py --devices 0 1 2 3 4 5 --num-epoch ${num_epoch} \
#     --dataset ${dataset} --root-path ${dgl_data_root} \
#     --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_6wk
python ${sgnn_dir}/train_${model}.py --num-worker 6 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage ${sgnn_cache_pct[6]} > ${log}.log 2> ${log}.err

log=${xgnn_log}_6wk
python ${sgnn_dir}/train_${model}.py --num-worker 6 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage ${xgnn_cache_pct[6]} > ${log}.log 2> ${log}.err

# # 7GPU
# log=${dgl_log}_7wk
# python ${dgl_dir}/train_${model}.py --devices 0 1 2 3 4 5 6 --num-epoch ${num_epoch} \
#     --dataset ${dataset} --root-path ${dgl_data_root} \
#     --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_7wk
python ${sgnn_dir}/train_${model}.py --num-worker 7 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage ${sgnn_cache_pct[7]} > ${log}.log 2> ${log}.err

log=${xgnn_log}_7wk
python ${sgnn_dir}/train_${model}.py --num-worker 7 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage ${xgnn_cache_pct[7]} > ${log}.log 2> ${log}.err

# # 8GPU
# log=${dgl_log}_8wk
# python ${dgl_dir}/train_${model}.py --devices 0 1 2 3 4 5 6 7 --num-epoch ${num_epoch} \
#     --dataset ${dataset} --root-path ${dgl_data_root} \
#     --use-gpu-sampling --use-uva-feat > ${log}.log 2> ${log}.err

log=${sgnn_log}_8wk
python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract \
    --cache-percentage ${sgnn_cache_pct[8]} > ${log}.log 2> ${log}.err

log=${xgnn_log}_8wk
python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph --cache-percentage ${xgnn_cache_pct[8]} > ${log}.log 2> ${log}.err

}


# pa gcn
scalability "papers100M" "pa" "gcn"

# uk graphsage
scalability "uk-2006-05" "uk" "graphsage"