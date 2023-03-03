#!/bin/bash 
set -x
export SAMGRAPH_HUGE_PAGE=1

dgl_dir=../../example/dgl/multi_gpu/
sgnn_dir=../../example/samgraph/sgnn/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run-logs/large_graph/${TIME_STAMPS}

dgl_devices="0 1 2 3 4 5 6 7"
num_worker=8
num_epoch=3

mkdir -p $log_dir

dataset="com-friendster"

# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.08 --unified-memory --unified-memory-percentage 0.0 1.0 > ${log_dir}/sgnn_gcn_cf_khop0_pipe.log 2> ${log_dir}/sgnn_gcn_cf_khop0_pipe.log.err
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.08 --unified-memory --unified-memory-percentage 0.0 1.0 > ${log_dir}/sgnn_gcn_cf_khop0_break.log 2> ${log_dir}/sgnn_gcn_cf_khop0_break.log.err

# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.01 > ${log_dir}/xgnn_gcn_cf_khop0_pipe.log 2> ${log_dir}/xgnn_gcn_cf_khop0_pipe.log.err
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.01 > ${log_dir}/xgnn_gcn_cf_khop0_break.log 2> ${log_dir}/xgnn_gcn_cf_khop0_break.log.err


dataset="uk-2006-05"

python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree  --empty-feat 24 --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.13 --unified-memory --unified-memory-percentage 0.0 1.0 > ${log_dir}/sgnn_gcn_uk_khop0_pipe.log 2> ${log_dir}/sgnn_gcn_uk_khop0_pipe.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree  --empty-feat 24 --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.13 --unified-memory --unified-memory-percentage 0.0 1.0 > ${log_dir}/sgnn_gcn_uk_khop0_break.log 2> ${log_dir}/sgnn_gcn_uk_khop0_break.log.err

python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.09 > ${log_dir}/xgnn_gcn_uk_khop0_pipe.log 2> ${log_dir}/xgnn_gcn_uk_khop0_pipe.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.09 > ${log_dir}/xgnn_gcn_uk_khop0_break.log 2> ${log_dir}/xgnn_gcn_uk_khop0_break.log.err






# for khop3
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --gpu-extract --cache-percentage 0.08 --unified-memory --unified-memory-percentage 0.0 1.0 > ${log_dir}/sgnn_gcn_cf_khop3_pipe.log 2> ${log_dir}/sgnn_gcn_cf_khop3_pipe.log.err
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --gpu-extract --cache-percentage 0.08 --unified-memory --unified-memory-percentage 0.0 1.0 > ${log_dir}/sgnn_gcn_cf_khop3_break.log 2> ${log_dir}/sgnn_gcn_cf_khop3_break.log.err
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.01 > ${log_dir}/xgnn_gcn_cf_khop3_pipe.log 2> ${log_dir}/xgnn_gcn_cf_khop3_pipe.log.err
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.01 > ${log_dir}/xgnn_gcn_cf_khop3_break.log 2> ${log_dir}/xgnn_gcn_cf_khop3_break.log.err
