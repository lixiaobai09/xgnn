#!/bin/bash -x
set -x
export SAMGRAPH_HUGE_PAGE=1
sgnn_dir=../../example/samgraph/sgnn/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run-logs/factor_analysis/${TIME_STAMPS}

num_worker=4
num_epoch=5

mkdir -p $log_dir

dataset="twitter"
# for gcn
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.09 > ${log_dir}/xgnn_gcn_tw_1.log 2> ${log_dir}/xgnn_gcn_tw_1.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --use-dist-graph 1.0 --gpu-extract --cache-percentage 0.19 > ${log_dir}/xgnn_gcn_tw_2.log 2> ${log_dir}/xgnn_gcn_tw_2.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.79 > ${log_dir}/xgnn_gcn_tw_3.log 2> ${log_dir}/xgnn_gcn_tw_3.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.79 > ${log_dir}/xgnn_gcn_tw_4.log 2> ${log_dir}/xgnn_gcn_tw_4.log.err

dataset="papers100M"
# for gcn
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.08 > ${log_dir}/xgnn_gcn_pa_1.log 2> ${log_dir}/xgnn_gcn_pa_1.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --use-dist-graph 1.0 --gpu-extract --cache-percentage 0.17 > ${log_dir}/xgnn_gcn_pa_2.log 2> ${log_dir}/xgnn_gcn_pa_2.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.69 > ${log_dir}/xgnn_gcn_pa_3.log 2> ${log_dir}/xgnn_gcn_pa_3.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.69 > ${log_dir}/xgnn_gcn_pa_4.log 2> ${log_dir}/xgnn_gcn_pa_4.log.err

dataset="uk-2006-05"
# for graphsage
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.01 > ${log_dir}/xgnn_graphsage_uk_1.log 2> ${log_dir}/xgnn_graphsage_uk_1.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --use-dist-graph 1.0 --gpu-extract --cache-percentage 0.12 > ${log_dir}/xgnn_graphsage_uk_2.log 2> ${log_dir}/xgnn_graphsage_uk_2.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.51 > ${log_dir}/xgnn_graphsage_uk_3.log 2> ${log_dir}/xgnn_graphsage_uk_3.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.51 > ${log_dir}/xgnn_graphsage_uk_4.log 2> ${log_dir}/xgnn_graphsage_uk_4.log.err

dataset="com-friendster"
# for graphsage
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.00 > ${log_dir}/xgnn_graphsage_cf_1.log 2> ${log_dir}/xgnn_graphsage_cf_1.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --use-dist-graph 1.0 --gpu-extract --cache-percentage 0.23 > ${log_dir}/xgnn_graphsage_cf_2.log 2> ${log_dir}/xgnn_graphsage_cf_2.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.93 > ${log_dir}/xgnn_graphsage_cf_3.log 2> ${log_dir}/xgnn_graphsage_cf_3.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.93 > ${log_dir}/xgnn_graphsage_cf_4.log 2> ${log_dir}/xgnn_graphsage_cf_4.log.err
