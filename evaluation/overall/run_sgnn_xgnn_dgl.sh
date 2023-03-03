#!/bin/bash -x
set -x
export SAMGRAPH_HUGE_PAGE=1

dgl_dir=../../example/dgl/multi_gpu/
sgnn_dir=../../example/samgraph/sgnn/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run-logs/overall/${TIME_STAMPS}

dgl_devices="0 1 2 3 4 5 6 7"
num_worker=8
num_epoch=10

mkdir -p $log_dir

dataset="twitter"
# for gcn
python ${dgl_dir}/train_gcn.py --batch-size 6000 --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_gcn_tw.log 2> ${log_dir}/dgl_gcn_tw.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.08 > ${log_dir}/sgnn_gcn_tw.log 2> ${log_dir}/sgnn_gcn_tw.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.19 > ${log_dir}/xgnn_gcn_tw.log 2> ${log_dir}/xgnn_gcn_tw.log.err
# # for graphsage
python ${dgl_dir}/train_graphsage.py --batch-size 6000 --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_graphsage_tw.log 2> ${log_dir}/dgl_graphsage_tw.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.17 > ${log_dir}/sgnn_graphsage_tw.log 2> ${log_dir}/sgnn_graphsage_tw.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.25 > ${log_dir}/xgnn_graphsage_tw.log 2> ${log_dir}/xgnn_graphsage_tw.log.err
# # for pinsage
python ${dgl_dir}/train_pinsage.py --batch-size 6000 --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_pinsage_tw.log 2> ${log_dir}/dgl_pinsage_tw.log.err
python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --gpu-extract --cache-percentage 0.11 > ${log_dir}/sgnn_pinsage_tw.log 2> ${log_dir}/sgnn_pinsage_tw.log.err
python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --gpu-extract --part-cache --use-dist-graph --cache-percentage 0.21 > ${log_dir}/xgnn_pinsage_tw.log 2> ${log_dir}/xgnn_pinsage_tw.log.err

dataset="papers100M"
# for gcn
python ${dgl_dir}/train_gcn.py --batch-size 6000 --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_gcn_pa.log 2> ${log_dir}/dgl_gcn_pa.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.07 > ${log_dir}/sgnn_gcn_pa.log 2> ${log_dir}/sgnn_gcn_pa.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.16 > ${log_dir}/xgnn_gcn_pa.log 2> ${log_dir}/xgnn_gcn_pa.log.err
# for graphsage
python ${dgl_dir}/train_graphsage.py --batch-size 6000 --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_graphsage_pa.log 2> ${log_dir}/dgl_graphsage_pa.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.11 > ${log_dir}/sgnn_graphsage_pa.log 2> ${log_dir}/sgnn_graphsage_pa.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.20 > ${log_dir}/xgnn_graphsage_pa.log 2> ${log_dir}/xgnn_graphsage_pa.log.err
# for pinsage
python ${dgl_dir}/train_pinsage.py --batch-size 6000 --devices ${dgl_devices} --num-epoch ${num_epoch} --dataset ${dataset} --use-gpu-sampling --use-uva-feat > ${log_dir}/dgl_pinsage_pa.log 2> ${log_dir}/dgl_pinsage_pa.log.err
python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --gpu-extract --cache-percentage 0.08 > ${log_dir}/sgnn_pinsage_pa.log 2> ${log_dir}/sgnn_pinsage_pa.log.err
python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --gpu-extract --part-cache --use-dist-graph --cache-percentage 0.17 > ${log_dir}/xgnn_pinsage_pa.log 2> ${log_dir}/xgnn_pinsage_pa.log.err

dataset="uk-2006-05"
# for gcn
# pipeline will OOM
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.0 > ${log_dir}/sgnn_gcn_uk.log 2> ${log_dir}/sgnn_gcn_uk.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.09 > ${log_dir}/xgnn_gcn_uk.log 2> ${log_dir}/xgnn_gcn_uk.log.err
# # for graphsage
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.01 > ${log_dir}/sgnn_graphsage_uk.log 2> ${log_dir}/sgnn_graphsage_uk.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.12 > ${log_dir}/xgnn_graphsage_uk.log 2> ${log_dir}/xgnn_graphsage_uk.log.err
# # for pinsage
# python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --gpu-extract --cache-percentage 0.0 > ${log_dir}/sgnn_pinsage_uk.log 2> ${log_dir}/sgnn_pinsage_uk.log.err
python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --gpu-extract --part-cache --use-dist-graph --cache-percentage 0.09 > ${log_dir}/xgnn_pinsage_uk.log 2> ${log_dir}/xgnn_pinsage_uk.log.err

dataset="com-friendster"
# for gcn
# python ${dgl_dir}/train_gcn.py --batch-size 6000 --devices ${dgl_devices} --num-epoch ${num_epoch} --root-path /data/samgraph --dataset ${dataset} --use-uva --use-uva-feat > ${log_dir}/dgl_gcn_cf.log
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.0 > ${log_dir}/sgnn_gcn_cf.log 2> ${log_dir}/sgnn_gcn_cf.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.0 > ${log_dir}/xgnn_gcn_cf.log 2> ${log_dir}/xgnn_gcn_cf.log.err
# for graphsage
# python ${dgl_dir}/train_graphsage.py --batch-size 6000 --devices ${dgl_devices} --num-epoch ${num_epoch} --root-path /data/samgraph --dataset ${dataset} --use-uva --use-uva-feat > ${log_dir}/dgl_graphsage_cf.log
# python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.0 > ${log_dir}/sgnn_graphsage_cf.log 2> ${log_dir}/sgnn_graphsage_cf.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.24 > ${log_dir}/xgnn_graphsage_cf.log 2> ${log_dir}/xgnn_graphsage_cf.log.err
# # for pinsage
# python ${dgl_dir}/train_pinsage.py --batch-size 6000 --devices ${dgl_devices} --num-epoch ${num_epoch} --root-path /data/samgraph --dataset ${dataset} --use-uva --use-uva-feat > ${log_dir}/dgl_pinsage_cf.log
# python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --gpu-extract --cache-percentage 0.0 > ${log_dir}/sgnn_pinsage_cf.log 2> ${log_dir}/sgnn_pinsage_cf.log.err
python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --gpu-extract --part-cache --use-dist-graph --cache-percentage 0.07 > ${log_dir}/xgnn_pinsage_cf.log 2> ${log_dir}/xgnn_pinsage_cf.log.err




# -----------------------
# breakdown
# -----------------------

dataset="twitter"
# for gcn
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.08 > ${log_dir}/sgnn_gcn_tw_break.log 2> ${log_dir}/sgnn_gcn_tw_break.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.19 > ${log_dir}/xgnn_gcn_tw_break.log 2> ${log_dir}/xgnn_gcn_tw_break.log.err
# for graphsage
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.17 > ${log_dir}/sgnn_graphsage_tw_break.log 2> ${log_dir}/sgnn_graphsage_tw_break.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.25 > ${log_dir}/xgnn_graphsage_tw_break.log 2> ${log_dir}/xgnn_graphsage_tw_break.log.err
# for pinsage
python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --gpu-extract --cache-percentage 0.11 > ${log_dir}/sgnn_pinsage_tw_break.log 2> ${log_dir}/sgnn_pinsage_tw_break.log.err
python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --gpu-extract --part-cache --use-dist-graph --cache-percentage 0.21 > ${log_dir}/xgnn_pinsage_tw_break.log 2> ${log_dir}/xgnn_pinsage_tw_break.log.err

dataset="papers100M"
# for gcn
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.07 > ${log_dir}/sgnn_gcn_pa_break.log 2> ${log_dir}/sgnn_gcn_pa_break.log.err
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.16 > ${log_dir}/xgnn_gcn_pa_break.log 2> ${log_dir}/xgnn_gcn_pa_break.log.err
# # for graphsage
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.11 > ${log_dir}/sgnn_graphsage_pa_break.log 2> ${log_dir}/sgnn_graphsage_pa_break.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.20 > ${log_dir}/xgnn_graphsage_pa_break.log 2> ${log_dir}/xgnn_graphsage_pa_break.log.err
# # for pinsage
python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --gpu-extract --cache-percentage 0.08 > ${log_dir}/sgnn_pinsage_pa_break.log 2> ${log_dir}/sgnn_pinsage_pa_break.log.err
python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --gpu-extract --part-cache --use-dist-graph --cache-percentage 0.17 > ${log_dir}/xgnn_pinsage_pa_break.log 2> ${log_dir}/xgnn_pinsage_pa_break.log.err

dataset="uk-2006-05"
# for gcn
# pipeline will OOM
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.09 > ${log_dir}/xgnn_gcn_uk_break.log 2> ${log_dir}/xgnn_gcn_uk_break.log.err
# # for graphsage
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.01 > ${log_dir}/sgnn_graphsage_uk_break.log 2> ${log_dir}/sgnn_graphsage_uk_break.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.12 > ${log_dir}/xgnn_graphsage_uk_break.log 2> ${log_dir}/xgnn_graphsage_uk_break.log.err
# # for pinsage
python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --gpu-extract --part-cache --use-dist-graph --cache-percentage 0.09 > ${log_dir}/xgnn_pinsage_uk_break.log 2> ${log_dir}/xgnn_pinsage_uk_break.log.err

dataset="com-friendster"
# for gcn
python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.0 > ${log_dir}/xgnn_gcn_cf_break.log 2> ${log_dir}/xgnn_gcn_cf_break.log.err
# for graphsage
python ${sgnn_dir}/train_graphsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.24 > ${log_dir}/xgnn_graphsage_cf_break.log 2> ${log_dir}/xgnn_graphsage_cf_break.log.err
# for pinsage
python ${sgnn_dir}/train_pinsage.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --gpu-extract --part-cache --use-dist-graph --cache-percentage 0.07 > ${log_dir}/xgnn_pinsage_cf_break.log 2> ${log_dir}/xgnn_pinsage_cf_break.log.err
