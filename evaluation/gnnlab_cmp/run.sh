#!/bin/bash -x
set -x
export SAMGRAPH_HUGE_PAGE=1

gnnlab_dir=../../example/samgraph/multi_gpu/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run-logs/gnnlab/${TIME_STAMPS}

num_worker=8
num_epoch=10

mkdir -p $log_dir



dataset="twitter"
# 8xGPU
python ${gnnlab_dir}/train_graphsage.py --dataset twitter --cache-policy pre_sample --cache-percentage 0.31 --batch-size 6000 --pipeline --sample-type khop2 --num-epoch ${num_epoch} --num-sample-worker 2 --num-train-worker 6 > ${log_dir}/gnnlab_graphsage_8g_tw.log 2> ${log_dir}/gnnlab_graphsage_8g_tw.log.err
# 4xGPU
python ${gnnlab_dir}/train_graphsage.py --dataset twitter --cache-policy pre_sample --cache-percentage 0.31 --batch-size 6000 --pipeline --sample-type khop2 --num-epoch ${num_epoch} --num-sample-worker 1 --num-train-worker 3 > ${log_dir}/gnnlab_graphsage_4g_tw.log 2> ${log_dir}/gnnlab_graphsage_4g_tw.log.err
# 1xGPU
python ${gnnlab_dir}/train_graphsage.py --dataset twitter --cache-policy pre_sample --cache-percentage 0.13 --batch-size 6000 --pipeline --sample-type khop2 --num-epoch ${num_epoch} --single-gpu > ${log_dir}/gnnlab_graphsage_1g_tw.log 2> ${log_dir}/gnnlab_graphsage_1g_tw.log.err
