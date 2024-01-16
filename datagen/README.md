# Datagen

This guide shows how to get the original datasets and convert them into the format that XGNN can read. We convert those data into binary format so that GNN systems can read the data very fast by using `MMAP`.



Default dataset path is `/graph-learning/samgraph/{dataset name}`.  `/graph-learning/samgraph/` is the default dataset root.

```bash
> tree -L 2 /graph-learning
/graph-learning
├── data-raw                    # original downloaded dataset  
│   ├── com-friendster
│   ├── com-friendster.tar.zst
│   ├── papers100M-bin
│   ├── papers100M-bin.zip
│   ├── twitter
│   └── uk-2006-05
└── samgraph                   # The converted dataset
    ├── com-friendster 
    ├── papers100M
    ├── twitter
    └── uk-2006-05
```



```bash
> tree /graph-learning/samgraph/papers100M
/graph-learning/samgraph/papers100M
├── cache_by_degree.bin      # vertexid sorted by cache rank(Higher rank, higher oppotunity to be cached)
├── feat.bin                 # vertex feature binary data
├── indices.bin              # csr indices stored as uint32
├── indptr.bin               # csr indptr stored as uint32
├── label.bin                # vertex label binary data
├── meta.txt                 # dataset meta data
├── test_set.bin             # testset node id list as uint32
├── train_set.bin            # trainset node id list as uint32
└── valid_set.bin            # validset node id list as uint32

0 directories, 9 files
```



## Disk Space Requirement

To store all four datasets, your disk should have at least **256GB** of free space to store dataset files.

```
12G     /graph-learning/samgraph/uk-2006-05
14G     /graph-learning/samgraph/com-friendster
61G     /graph-learning/samgraph/papers100M
5.8G    /graph-learning/samgraph/twitter
```

```
23G     /graph-learning/data-raw/uk-2006-05
14G     /graph-learning/data-raw/twitter
57G     /graph-learning/data-raw/papers100M-bin
45G     /graph-learning/data-raw/com-friendster
```


## 1. Download And Convert

Create the dataset directory:

```bash
sudo mkdir -p /graph-learning/samgraph
sudo mkdir -p /graph-learning/data-raw
sudo chmod -R 777 /graph-learning
```



Download the dataset and convert them into binary format:

```bash
cd xgnn/datagen

python friendster.py
python papers100M.py
bash twitter.sh
bash uk-2006-05.sh
```



Now we have:

```bash
> tree /graph-learning/samgraph/papers100M
/graph-learning/samgraph/papers100M
├── feat.bin
├── indices.bin
├── indptr.bin
├── label.bin
├── meta.txt
├── test_set.bin
├── train_set.bin
└── valid_set.bin
```

## 2. Generate Cache Rank Table

The degree-based cache policy uses the out-degree as cache rank. The ranking only needs to be preprocessed once. The cache rank table is a vertex-id list sorted by their out-degree.

```bash
cd xgnn/utility/data-process
mkdir build
cd build
cmake ..

make cache-by-degree -j

# degree-based cache policy
./cache-by-degree -g products
./cache-by-degree -g papers100M
./cache-by-degree -g twitter
./cache-by-degree -g uk-2006-05

Now we have:

```bash
/graph-learning/samgraph/papers100M
├── cache_by_degree.bin   # new added
├── feat.bin
├── indices.bin
├── indptr.bin
├── label.bin
├── meta.txt
├── test_set.bin
├── train_set.bin
└── valid_set.bin
```