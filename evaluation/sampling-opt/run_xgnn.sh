# comm
MY_DIR="$(dirname $0)"
TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")

dgl_dir=${MY_DIR}/../../example/dgl/multi_gpu/
sgnn_dir=${MY_DIR}/../../example/samgraph/sgnn/

export SAMGRAPH_HUGE_PAGE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# config
log_dir=${MY_DIR}/run-logs/sample-time/${TIME_STAMPS}
model=gcn
num_worker=4
num_epoch=3

dgl_log=${log_dir}/dgl_${model}
sgnn_log=${log_dir}/sgnn_${model}
xgnn_log=${log_dir}/xgnn_${model}

mkdir -p ${log_dir}


### twitter ###
sample_type=khop0
log=${xgnn_log}_tw_${sample_type}_${num_worker}wk
python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset twitter --sample-type ${sample_type} \
    --use-dist-graph 1 --cache-percentage 0.20 > ${log}.log 2> ${log}.err

sample_type=khop3
log=${xgnn_log}_tw_${sample_type}_${num_worker}wk
python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset twitter --sample-type ${sample_type} \
    --use-dist-graph 1 --cache-percentage 0.20 > ${log}.log 2> ${log}.err


### papers100M ###
# sample_type=khop0
# log=${xgnn_log}_pa_${sample_type}_${num_worker}wk
# python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
#     --num-epoch ${num_epoch} --dataset papers100M --sample-type ${sample_type} \
#     --use-dist-graph 1 --cache-percentage 0.15 > ${log}.log 2> ${log}.err

# sample_type=khop3
# log=${xgnn_log}_pa_${sample_type}_${num_worker}wk
# python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
#     --num-epoch ${num_epoch} --dataset papers100M --sample-type ${sample_type} \
#     --use-dist-graph 1 --cache-percentage 0.15 > ${log}.log 2> ${log}.err


### com-friendster ###
sample_type=khop0
log=${xgnn_log}_cf_${sample_type}_${num_worker}wk
python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset com-friendster --sample-type ${sample_type} \
    --use-dist-graph 1 --cache-percentage 0 > ${log}.log 2> ${log}.err

sample_type=khop3
log=${xgnn_log}_cf_${sample_type}_${num_worker}wk
python ${sgnn_dir}/train_${model}.py --num-worker ${num_worker} --cache-policy degree --batch-size 6000 \
    --num-epoch ${num_epoch} --dataset com-friendster --sample-type ${sample_type} \
    --use-dist-graph 1 --cache-percentage 0 > ${log}.log 2> ${log}.err

