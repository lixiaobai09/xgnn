#!/bin/bash 
set -x
export SAMGRAPH_HUGE_PAGE=1

dgl_dir=../../example/dgl/multi_gpu/
sgnn_dir=../../example/samgraph/sgnn/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run-logs/acc_test/${TIME_STAMPS}

dgl_devices="0 1 2 3 4 5 6 7"
num_worker=8

mkdir -p $log_dir

dataset="papers100M"
num_epoch=200

# papers100M acc: 56%
python ${dgl_dir}/train_graphsage.py --dataset papers100M --pipelining --report-acc 151 --num-epoch ${num_epoch} --use-gpu-sampling --use-uva-feat --devices ${dgl_devices} > ${log_dir}/dgl_graphsage_pa.log 2>&1
# python ${sgnn_dir}/train_graphsage.py --report-acc 151 --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.11 > ${log_dir}/sgnn_graphsage_pa.log 2>&1
python ${sgnn_dir}/train_graphsage.py --report-acc 151 --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.20 > ${log_dir}/xgnn_graphsage_pa.log 2>&1
