#!/bin/bash -x
set -x
export SAMGRAPH_HUGE_PAGE=1
sgnn_dir=../../example/samgraph/sgnn/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run-logs/${TIME_STAMPS}

num_epoch=10

mkdir -p $log_dir

dataset="com-friendster"
# for graphsage
CUDA_VISIBLE_DEVICES=0,1,2,3 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.93 > ${log_dir}/xgnn_graphsage_cf_4g_1_break.log 2> ${log_dir}/xgnn_graphsage_cf_4g_1_break.log.err
CUDA_VISIBLE_DEVICES=0,1,6,7 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.26 > ${log_dir}/xgnn_graphsage_cf_4g_2_break.log 2> ${log_dir}/xgnn_graphsage_cf_4g_2_break.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker 8 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.93 > ${log_dir}/xgnn_graphsage_cf_8g_break.log 2> ${log_dir}/xgnn_graphsage_cf_8g_break.log.err

CUDA_VISIBLE_DEVICES=0,1,2,3 python ${sgnn_dir}/train_graphsage.py --pipeline --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.93 > ${log_dir}/xgnn_graphsage_cf_4g_1.log 2> ${log_dir}/xgnn_graphsage_cf_4g_1.log.err
CUDA_VISIBLE_DEVICES=0,1,6,7 python ${sgnn_dir}/train_graphsage.py --pipeline --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.26 > ${log_dir}/xgnn_graphsage_cf_4g_2.log 2> ${log_dir}/xgnn_graphsage_cf_4g_2.log.err
python ${sgnn_dir}/train_graphsage.py --pipeline --num-worker 8 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.93 > ${log_dir}/xgnn_graphsage_cf_8g.log 2> ${log_dir}/xgnn_graphsage_cf_8g.log.err
