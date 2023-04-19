#!/bin/bash 
MY_DIR="$(dirname $0)"
TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")

dgl_dir=${MY_DIR}/../../example/dgl/multi_gpu/
sgnn_dir=${MY_DIR}/../../example/samgraph/sgnn/

export SAMGRAPH_HUGE_PAGE=1

set -x

log_dir=${MY_DIR}/run-logs/${TIME_STAMPS}
num_epoch=10

mkdir -p "$log_dir"


### graphsage cf ###
export CUDA_VISIBLE_DEVICES="0"
num_worker=1
dgl_devices="0"
model=graphsage
dataset="com-friendster"

# dgl
python ${dgl_dir}/train_${model}.py --batch-size 6000 --devices ${dgl_devices} \
    --num-epoch ${num_epoch} --root-path /data/samgraph --dataset ${dataset} \
    --use-uva --use-uva-feat \
    > ${log_dir}/dgl_${model}_cf_1wk.log 2> ${log_dir}/dgl_${model}_cf_1wk.err

# xgnn-f
python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract \
    --unified-memory --unified-memory-percentage 0 1 \
    --cache-percentage 0.33 \
    > ${log_dir}/xgnn-f_${model}_cf_1wk.log 2> ${log_dir}/xgnn-f_${model}_cf_1wk.err

python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
    --gpu-extract \
    --unified-memory --unified-memory-percentage 0 1 \
    --cache-percentage 0.33 \
    > ${log_dir}/xgnn-f_${model}_cf_1wk_bk.log 2> ${log_dir}/xgnn-f_${model}_cf_1wk_bk.err

# xgnn-g
python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
    --gpu-extract \
    --use-dist-graph 0.85 \
    --cache-percentage 0 \
    > ${log_dir}/xgnn-g_${model}_cf_1wk.log 2> ${log_dir}/xgnn-g_${model}_cf_1wk.err

python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
    --gpu-extract \
    --use-dist-graph 0.85 \
    --cache-percentage 0 \
    > ${log_dir}/xgnn-g_${model}_cf_1wk_bk.log 2> ${log_dir}/xgnn-g_${model}_cf_1wk_bk.err


# ### gcn uk ###
# export CUDA_VISIBLE_DEVICES="0,1"
# num_worker=2
# dgl_devices="0 1"
# model=gcn
# dataset="uk-2006-05"

# # dgl
# python ${dgl_dir}/train_${model}.py --batch-size 6000 --devices ${dgl_devices} \
#     --num-epoch ${num_epoch} --root-path /data/samgraph --dataset ${dataset} \
#     --use-uva --use-uva-feat \
#     > ${log_dir}/dgl_${model}_uk_2wk.log 2> ${log_dir}/dgl_${model}_uk_2wk.err

# # xgnn-f
# python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
#     --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
#     --gpu-extract \
#     --unified-memory --unified-memory-percentage 0 1 \
#     --cache-percentage 0.29 \
#     > ${log_dir}/xgnn-f_${model}_uk_2wk.log 2> ${log_dir}/xgnn-f_${model}_uk_2wk.err

# python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
#     --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
#     --gpu-extract \
#     --unified-memory --unified-memory-percentage 0 1 \
#     --cache-percentage 0 \
#     > ${log_dir}/xgnn-f_${model}_uk_2wk_bk.log 2> ${log_dir}/xgnn-f_${model}_uk_2wk_bk.err

# # xgnn-g
# python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
#     --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache \
#     --gpu-extract \
#     --use-dist-graph 1 \
#     --cache-percentage 0 \
#     > ${log_dir}/xgnn-g_${model}_uk_2wk.log 2> ${log_dir}/xgnn-g_${model}_uk_2wk.err

# python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
#     --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache \
#     --gpu-extract \
#     --use-dist-graph 1 \
#     --cache-percentage 0 \
#     > ${log_dir}/xgnn-g_gcn_uk_2wk_bk.log 2> ${log_dir}/xgnn-g_gcn_uk_2wk_bk.err



# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.08 --unified-memory --unified-memory-percentage 0.0 1.0 > ${log_dir}/sgnn_gcn_cf_khop0_pipe.log 2> ${log_dir}/sgnn_gcn_cf_khop0_pipe.log.err
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.08 --unified-memory --unified-memory-percentage 0.0 1.0 > ${log_dir}/sgnn_gcn_cf_khop0_break.log 2> ${log_dir}/sgnn_gcn_cf_khop0_break.log.err

# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.01 > ${log_dir}/xgnn_gcn_cf_khop0_pipe.log 2> ${log_dir}/xgnn_gcn_cf_khop0_pipe.log.err
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.01 > ${log_dir}/xgnn_gcn_cf_khop0_break.log 2> ${log_dir}/xgnn_gcn_cf_khop0_break.log.err


# dataset="uk-2006-05"

# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree  --empty-feat 24 --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --gpu-extract --cache-percentage 0.13 --unified-memory --unified-memory-percentage 0.0 1.0 > ${log_dir}/sgnn_gcn_uk_khop0_pipe.log 2> ${log_dir}/sgnn_gcn_uk_khop0_pipe.log.err
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree  --empty-feat 24 --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --gpu-extract --cache-percentage 0.13 --unified-memory --unified-memory-percentage 0.0 1.0 > ${log_dir}/sgnn_gcn_uk_khop0_break.log 2> ${log_dir}/sgnn_gcn_uk_khop0_break.log.err

# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop0 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.09 > ${log_dir}/xgnn_gcn_uk_khop0_pipe.log 2> ${log_dir}/xgnn_gcn_uk_khop0_pipe.log.err
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop0 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.09 > ${log_dir}/xgnn_gcn_uk_khop0_break.log 2> ${log_dir}/xgnn_gcn_uk_khop0_break.log.err






# for khop3
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --gpu-extract --cache-percentage 0.08 --unified-memory --unified-memory-percentage 0.0 1.0 > ${log_dir}/sgnn_gcn_cf_khop3_pipe.log 2> ${log_dir}/sgnn_gcn_cf_khop3_pipe.log.err
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --gpu-extract --cache-percentage 0.08 --unified-memory --unified-memory-percentage 0.0 1.0 > ${log_dir}/sgnn_gcn_cf_khop3_break.log 2> ${log_dir}/sgnn_gcn_cf_khop3_break.log.err
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --pipeline --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.01 > ${log_dir}/xgnn_gcn_cf_khop3_pipe.log 2> ${log_dir}/xgnn_gcn_cf_khop3_pipe.log.err
# python ${sgnn_dir}/train_gcn.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph --cache-percentage 0.01 > ${log_dir}/xgnn_gcn_cf_khop3_break.log 2> ${log_dir}/xgnn_gcn_cf_khop3_break.log.err
