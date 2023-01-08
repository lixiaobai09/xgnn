#!/bin/bash 
dgl_dir=../../example/dgl/multi_gpu/
sgnn_dir=../../example/samgraph/sgnn/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run-logs/overall/${TIME_STAMPS}

dgl_devices="0 1 2 3 4 5 6 7"
num_worker=8
num_epoch=3

mkdir -p $log_dir

# TODO: get cache percentage
dataset="twitter"
# for gcn
python ${dgl_dir}/train_gcn.py --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_gcn_tw.log
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.07 > ${log_dir}/sgnn_gcn_tw.log
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.16 > ${log_dir}/xgnn_gcn_tw.log
# for graphsage
python ${dgl_dir}/train_graphsage.py --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_graphsage_tw.log
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.11 > ${log_dir}/sgnn_graphsage_tw.log
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.20 > ${log_dir}/xgnn_graphsage_tw.log

dataset="papers100M"
# for gcn
python ${dgl_dir}/train_gcn.py --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_gcn_pa.log
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.07 > ${log_dir}/sgnn_gcn_pa.log
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.16 > ${log_dir}/xgnn_gcn_pa.log
# for graphsage
python ${dgl_dir}/train_graphsage.py --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_graphsage_pa.log
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.11 > ${log_dir}/sgnn_graphsage_pa.log
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.20 > ${log_dir}/xgnn_graphsage_pa.log

# TODO: get cache percentage
dataset="uk-2006-05"
# for gcn
python ${dgl_dir}/train_gcn.py --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_gcn_uk.log
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.07 > ${log_dir}/sgnn_gcn_uk.log
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.16 > ${log_dir}/xgnn_gcn_uk.log
# for graphsage
python ${dgl_dir}/train_graphsage.py --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_graphsage_uk.log
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.11 > ${log_dir}/sgnn_graphsage_uk.log
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.20 > ${log_dir}/xgnn_graphsage_uk.log

# TODO: get cache percentage
dataset="com-friendster"
# for gcn
python ${dgl_dir}/train_gcn.py --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_gcn_cf.log
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.07 > ${log_dir}/sgnn_gcn_cf.log
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.16 > ${log_dir}/xgnn_gcn_cf.log
# for graphsage
python ${dgl_dir}/train_graphsage.py --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_graphsage_cf.log
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.11 > ${log_dir}/sgnn_graphsage_cf.log
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.20 > ${log_dir}/xgnn_graphsage_cf.log

