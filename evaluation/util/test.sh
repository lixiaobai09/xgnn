python ../../example/samgraph/sgnn/train_gcn.py --use-dist-graph 1 \
    --part-cache --gpu-extract --cache-policy degree --cache-percentage 0  \
    --batch-size 6000  -ll info --num-epoch 3 --dataset products  --num-worker 4 \
    --sample-type khop3 -pl 3 --pipeline 