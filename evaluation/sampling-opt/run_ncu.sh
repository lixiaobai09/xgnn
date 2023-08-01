# comm
MY_DIR="$(dirname $0)"
TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")

dgl_dir=${MY_DIR}/../../example/dgl/multi_gpu/
sgnn_dir=${MY_DIR}/../../example/samgraph/sgnn/

export SAMGRAPH_HUGE_PAGE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# config
log_dir=${MY_DIR}/run-logs/ncu-prof/${TIME_STAMPS}
model=gcn
num_epoch=3

dgl_log=${log_dir}/dgl_${model}_${ds_short}
sgnn_log=${log_dir}/sgnn_${model}_${ds_short}
xgnn_log=${log_dir}/xgnn_${model}_${ds_short}

mkdir -p ${log_dir}

# setup
profiler="sudo /tmp/var/target/linux-desktop-glibc_2_11_3-x64/ncu"
prof_arg="--target-processes all --replay-mode kernel"

declare -A ds_short=(
    ["com-friendster"]="cf" ["papers100M"]="pa" ["uk-2006-05"]="uk" ["twitter"]="tw"

)

# input: dataset, sample_type, sample_kernel, num_worker
xgnn_gcn_nvltx() {
    local dataset=$1
    local sample_type=$2
    local sample_kernel=$3
    local num_worker=$4
    metrics=nvlrx__bytes.sum,nvlrx__bytes_data_protocol.sum,nvlrx__bytes_data_user.sum

    local log=${ds_short[$dataset]}-${sample_type}-${num_worker}wk
    cmd="$profiler $prof_arg --log-file $log_dir/$log.ncu \
        --kernel-name $sample_kernel --metrics $metrics \
        `which python` $sgnn_dir/train_$model.py --num-worker $num_worker \
        --cache-policy degree --batch-size 6000 --num-epoch $num_epoch --dataset $dataset \
        --sample-type $sample_type --use-dist-graph 1 --cache-percentage 0 \
        > $log_dir/$log.log 2> $log_dir/$log.err"

    echo ">>> $cmd"
    eval "$cmd"
    sleep 1
}

# tw
xgnn_gcn_nvltx "twitter" "khop0" "sample_khop0" "4"
xgnn_gcn_nvltx "twitter" "khop3" "sample_khop3" "4"

# pa
# xgnn_gcn_nvltx "papers100M" "khop0" "sample_khop0" "4"
# xgnn_gcn_nvltx "papers100M" "khop3" "sample_khop3" "4"

# # uk
# xgnn_gcn_nvltx "uk-2006-05" "khop0" "sample_khop0"
# xgnn_gcn_nvltx "uk-2006-05" "khop3" "sample_khop3"

# # cf
xgnn_gcn_nvltx "com-friendster" "khop0" "sample_khop0" "4"
xgnn_gcn_nvltx "com-friendster" "khop3" "sample_khop3" "4"
