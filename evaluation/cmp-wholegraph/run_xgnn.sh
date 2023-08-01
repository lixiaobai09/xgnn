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


mkdir -p "$log_dir"


# #############  4xV100  #############
# # gcn+pr
# dataset=products
# log=${log_dir}/xgnn_gcn_pr
# python ${sgnn_dir}/train_gcn.py --num-worker 4 --cache-policy degree --batch-size 6000 \
#     --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
#     --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err

# log=${log_dir}/xgnn_gcn_pr_bk
# python ${sgnn_dir}/train_gcn.py --num-worker 4 --cache-policy degree --batch-size 6000 \
#     --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
#     --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err


# # graphsage + tw
# dataset=twitter
# log=${log_dir}/xgnn_graphsage_tw
# python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 \
#     --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
#     --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err

# log=${log_dir}/xgnn_graphsage_tw_bk
# python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 \
#     --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
#     --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err



#############  8xA100  #############
# gcn+tw
dataset=twitter
log=${log_dir}/xgnn_gcn_tw
python ${sgnn_dir}/train_gcn.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err

log=${log_dir}/xgnn_gcn_tw_bk
python ${sgnn_dir}/train_gcn.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err


# gcn+pa
dataset=papers100M
log=${log_dir}/xgnn_gcn_pa
python ${sgnn_dir}/train_gcn.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err

log=${log_dir}/xgnn_gcn_pa_bk
python ${sgnn_dir}/train_gcn.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err


# graphsage+uk
dataset=uk-2006-05
log=${log_dir}/xgnn_graphsage_uk
python ${sgnn_dir}/train_graphsage.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err

log=${log_dir}/xgnn_graphsage_uk_bk
python ${sgnn_dir}/train_graphsage.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err


# graphsage+cf
dataset=com-friendster
log=${log_dir}/xgnn_graphsage_cf
python ${sgnn_dir}/train_graphsage.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err

log=${log_dir}/xgnn_graphsage_cf_bk
python ${sgnn_dir}/train_graphsage.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 1 > ${log}.log 2> ${log}.err