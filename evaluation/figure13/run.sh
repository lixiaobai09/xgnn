#!/bin/bash -x
set -x
export SAMGRAPH_HUGE_PAGE=1
sgnn_dir=../../example/samgraph/sgnn/

TIME_STAMPS=$(date "+%Y-%m-%d_%H-%M-%S")
log_dir=./run-logs/${TIME_STAMPS}

num_epoch=10

mkdir -p $log_dir


dataset="com-friendster"
# for xgnn 
CUDA_VISIBLE_DEVICES=0,1,2,3 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.93 > ${log_dir}/xgnn_graphsage_cf_4g_1_origin_break.log 2> ${log_dir}/xgnn_graphsage_cf_4g_1_origin_break.log.err
CUDA_VISIBLE_DEVICES=0,1,6,7 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.26 > ${log_dir}/xgnn_graphsage_cf_4g_2_origin_break.log 2> ${log_dir}/xgnn_graphsage_cf_4g_2_origin_break.log.err
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python ${sgnn_dir}/train_graphsage.py --num-worker 6 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.59 > ${log_dir}/xgnn_graphsage_cf_6g_origin_break.log 2> ${log_dir}/xgnn_graphsage_cf_6g_origin_break.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker 8 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.93 > ${log_dir}/xgnn_graphsage_cf_8g_origin_break.log 2> ${log_dir}/xgnn_graphsage_cf_8g_origin_break.log.err

# for clique
export SAMGRAPH_CLIQUE=1
CUDA_VISIBLE_DEVICES=0,1,2,3 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.93 > ${log_dir}/xgnn_graphsage_cf_4g_1_clique_break.log 2> ${log_dir}/xgnn_graphsage_cf_4g_1_clique_break.log.err
CUDA_VISIBLE_DEVICES=0,1,6,7 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.26 > ${log_dir}/xgnn_graphsage_cf_4g_2_clique_break.log 2> ${log_dir}/xgnn_graphsage_cf_4g_2_clique_break.log.err
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python ${sgnn_dir}/train_graphsage.py --num-worker 6 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.41 > ${log_dir}/xgnn_graphsage_cf_6g_clique_break.log 2> ${log_dir}/xgnn_graphsage_cf_6g_clique_break.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker 8 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.93 > ${log_dir}/xgnn_graphsage_cf_8g_clique_break.log 2> ${log_dir}/xgnn_graphsage_cf_8g_clique_break.log.err
unset SAMGRAPH_CLIQUE

# for ICS22 solver
CUDA_VISIBLE_DEVICES=0,1,2,3 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --use-ics22-song-solver --clique-size 4 --ics22-song-alpha 0.01 --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.2325 > ${log_dir}/xgnn_graphsage_cf_4g_1_ics22_break.log 2> ${log_dir}/xgnn_graphsage_cf_4g_1_ics22_break.log.err
CUDA_VISIBLE_DEVICES=0,1,6,7 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --use-ics22-song-solver --clique-size 2 --ics22-song-alpha 0.01 --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.13 > ${log_dir}/xgnn_graphsage_cf_4g_2_ics22_break.log 2> ${log_dir}/xgnn_graphsage_cf_4g_2_ics22_break.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker 6 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --use-ics22-song-solver --clique-size 2 --ics22-song-alpha 0.005 --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.205 > ${log_dir}/xgnn_graphsage_cf_6g_ics22_break.log 2> ${log_dir}/xgnn_graphsage_cf_6g_ics22_break.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker 8 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --use-ics22-song-solver --clique-size 4 --ics22-song-alpha 0.01 --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.2325 > ${log_dir}/xgnn_graphsage_cf_8g_ics22_break.log 2> ${log_dir}/xgnn_graphsage_cf_8g_ics22_break.log.err




dataset="twitter"
# for xgnn 
CUDA_VISIBLE_DEVICES=0,1,2,3 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 1.0 > ${log_dir}/xgnn_graphsage_tw_4g_1_origin_break.log 2> ${log_dir}/xgnn_graphsage_tw_4g_1_origin_break.log.err
CUDA_VISIBLE_DEVICES=0,1,6,7 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.46 > ${log_dir}/xgnn_graphsage_tw_4g_2_origin_break.log 2> ${log_dir}/xgnn_graphsage_tw_4g_2_origin_break.log.err
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python ${sgnn_dir}/train_graphsage.py --num-worker 6 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.76 > ${log_dir}/xgnn_graphsage_tw_6g_origin_break.log 2> ${log_dir}/xgnn_graphsage_tw_6g_origin_break.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker 8 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 1.0 > ${log_dir}/xgnn_graphsage_tw_8g_origin_break.log 2> ${log_dir}/xgnn_graphsage_tw_8g_origin_break.log.err

# for clique
export SAMGRAPH_CLIQUE=1
CUDA_VISIBLE_DEVICES=0,1,2,3 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 1.0 > ${log_dir}/xgnn_graphsage_tw_4g_1_clique_break.log 2> ${log_dir}/xgnn_graphsage_tw_4g_1_clique_break.log.err
CUDA_VISIBLE_DEVICES=0,1,6,7 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.46 > ${log_dir}/xgnn_graphsage_tw_4g_2_clique_break.log 2> ${log_dir}/xgnn_graphsage_tw_4g_2_clique_break.log.err
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python ${sgnn_dir}/train_graphsage.py --num-worker 6 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.46 > ${log_dir}/xgnn_graphsage_tw_6g_clique_break.log 2> ${log_dir}/xgnn_graphsage_tw_6g_clique_break.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker 8 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 1.0 > ${log_dir}/xgnn_graphsage_tw_8g_clique_break.log 2> ${log_dir}/xgnn_graphsage_tw_8g_clique_break.log.err
unset SAMGRAPH_CLIQUE

# for ICS22 solver
CUDA_VISIBLE_DEVICES=0,1,2,3 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --use-ics22-song-solver --clique-size 4 --ics22-song-alpha 0.01 --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.25 > ${log_dir}/xgnn_graphsage_tw_4g_1_ics22_break.log 2> ${log_dir}/xgnn_graphsage_tw_4g_1_ics22_break.log.err
CUDA_VISIBLE_DEVICES=0,1,6,7 python ${sgnn_dir}/train_graphsage.py --num-worker 4 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --use-ics22-song-solver --clique-size 2 --ics22-song-alpha 0.03 --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.23 > ${log_dir}/xgnn_graphsage_tw_4g_2_ics22_break.log 2> ${log_dir}/xgnn_graphsage_tw_4g_2_ics22_break.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker 6 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --use-ics22-song-solver --clique-size 2 --ics22-song-alpha 0.20 --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.26 > ${log_dir}/xgnn_graphsage_tw_6g_ics22_break.log 2> ${log_dir}/xgnn_graphsage_tw_6g_ics22_break.log.err
python ${sgnn_dir}/train_graphsage.py --num-worker 8 --cache-policy degree --batch-size 6000 --num-epoch ${num_epoch} --dataset ${dataset} --sample-type khop3 --use-ics22-song-solver --clique-size 4 --ics22-song-alpha 0.01 --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.25 > ${log_dir}/xgnn_graphsage_tw_8g_ics22_break.log 2> ${log_dir}/xgnn_graphsage_tw_8g_ics22_break.log.err


# parse the results
python parse_res.py -d ${log_dir} --dataset tw > placement_solver_tw.res
python parse_res.py -d ${log_dir} --dataset cf > placement_solver_cf.res


# plot the figures, output files are placement-solver-tw-fig-a.eps and placement-solver-cf-fig-b.eps
gnuplot figure_a.plt
gnuplot figure_b.plt
