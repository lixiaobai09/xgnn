#!/bin/bash 
set -x
export SAMGRAPH_HUGE_PAGE=1

dgl_dir=../../example/dgl/multi_gpu/
sgnn_dir=../../example/samgraph/sgnn/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run-logs/optimization/${TIME_STAMPS}

num_worker=4
num_epoch=3

mkdir -p $log_dir

dataset="uk-2006-05"
python ${sgnn_dir}/train_graphsage.py  --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.01 > ${log_dir}/${dataset}_graphsage_khop0.log 2> ${log_dir}/${dataset}_graphsage_khop0.log.err
python ${sgnn_dir}/train_graphsage.py  --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.01 --part-cache > ${log_dir}/${dataset}_graphsage_khop0_pcache.log 2> ${log_dir}/${dataset}_graphsage_khop0_pcache.log.err
python ${sgnn_dir}/train_graphsage.py  --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.12 --part-cache --use-dist-graph > ${log_dir}/${dataset}_graphsage_khop0_pcache_pgraph.log 2> ${log_dir}/${dataset}_graphsage_khop0_pcache_pgraph.log.err
python ${sgnn_dir}/train_graphsage.py  --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --gpu-extract --cache-percentage 0.12 --part-cache --use-dist-graph > ${log_dir}/${dataset}_graphsage_khop3_pcache_pgraph.log 2> ${log_dir}/${dataset}_graphsage_khop3_pcache_pgraph.log.err

dataset="twitter"
python ${sgnn_dir}/train_gcn.py  --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.06 > ${log_dir}/${dataset}_gcn_khop0.log 2> ${log_dir}/${dataset}_gcn_khop0.log.err
python ${sgnn_dir}/train_gcn.py  --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.06 --part-cache > ${log_dir}/${dataset}_gcn_khop0_pcache.log 2> ${log_dir}/${dataset}_gcn_khop0_pcache.log.err
python ${sgnn_dir}/train_gcn.py  --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.17 --part-cache --use-dist-graph > ${log_dir}/${dataset}_gcn_khop0_pcache_pgraph.log 2> ${log_dir}/${dataset}_gcn_khop0_pcache_pgraph.log.err
python ${sgnn_dir}/train_gcn.py  --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --gpu-extract --cache-percentage 0.17 --part-cache --use-dist-graph > ${log_dir}/${dataset}_gcn_khop3_pcache_pgraph.log 2> ${log_dir}/${dataset}_gcn_khop3_pcache_pgraph.log.err
