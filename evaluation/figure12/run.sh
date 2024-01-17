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
dataset=uk-2006-05
ds_short=uk
model=graphsage
num_epoch=10

dgl_log=${log_dir}/dgl_${model}_${ds_short}
sgnn_log=${log_dir}/sgnn_${model}_${ds_short}
xgnn_log=${log_dir}/xgnn_${model}_${ds_short}

mkdir -p ${log_dir}

xgnn_uk_graphsage_cache_pct=(-1 0.01 0.18 0.51 0.51)

### 1GPU ###
log=${xgnn_log}_1wk
python ${sgnn_dir}/train_${model}.py --num-worker 1 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 0.01 > ${log}.log 2> ${log}.err
python ${sgnn_dir}/train_${model}.py --num-worker 1 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 0.01 > ${log}-bk.log 2> ${log}-bk.err


### 2GPU ###
log=${xgnn_log}_2wk
python ${sgnn_dir}/train_${model}.py --num-worker 2 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 0.18 > ${log}.log 2> ${log}.err
python ${sgnn_dir}/train_${model}.py --num-worker 2 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 0.18 > ${log}-bk.log 2> ${log}-bk.err


### 4GPU ###
log=${xgnn_log}_4wk
python ${sgnn_dir}/train_${model}.py --num-worker 4 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 0.51 > ${log}.log 2> ${log}.err
python ${sgnn_dir}/train_${model}.py --num-worker 4 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 0.51 > ${log}-bk.log 2> ${log}-bk.err


### 8GPU ###
log=${xgnn_log}_8wk
python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 0.51 > ${log}.log 2> ${log}.err
python ${sgnn_dir}/train_${model}.py --num-worker 8 --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
    --gpu-extract --use-dist-graph 1 --cache-percentage 0.51 > ${log}-bk.log 2> ${log}-bk.err


# parse the results
python parse_res.py -d ${log_dir} > memory-usage.dat

# plot the figure, the output file is "memory-usage.eps"
gnuplot figure.plt
