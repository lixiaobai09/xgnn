# XGNN

XGNN is a multi-GPU GNN training system that fully utilizes system memory (e.g., GPU and host memory) and high-speed interconnects. The Global GNN Memory Store (GGMS) is the core design of XGNN, which abstracts underlying resources to provide a unified memory store for GNN training. It partitions hybrid input data, including graph topological and feature data, across both GPU and host memory. GGMS also provides easy-to-use APIs for GNN applications to access data transparently, forwarding data access requests to the actual physical data partitions automatically.

## Terminology
SamGraph is the framework shared by the above system. SGNN is the initial version of DGL+C, a baseline system. 

## Table of Contents
  - [Project Structure](#project-structure)
  - [Paper's Hardware Configuration](#papers-hardware-configuration)
  - [Installation](#installation)
    - [Software Version](#software-version)
    - [Install CUDA11.7](#install-cuda117)
    - [Install GNN Training Systems](#install-gnn-training-systems)
    - [Change ULIMIT and Open THP](#change-ulimit-and-open-transparent-huge-pages)
  - [Dataset Preprocessing](#dataset-preprocessing)
  - [QuickStart: Use XGNN to train GNN models](#quickstart-use-xgnn-to-train-gnn-models)
  - [Experiments](#experiments)
  - [License](#license)


## Project Structure

```bash
> tree .
├── datagen                     # Dataset Preprocessing
├── example
│   ├── dgl
│   │   ├── multi_gpu           # DGL models
│   ├── samgraph
│   │   ├── sgnn                # SGNN models
├── evaluation                  # Experiment Scripts
│   ├── overall
│   ├── factor_analysis
├── samgraph                    # XGNN, SGNN source codes
└── utility                     # Useful tools for dataset preprocessing
```



## Paper's Hardware Configuration
XGNN aims to accelerate GNN training at multi-GPU platforms with high-speed interconnects (like NVLink).
In our evaluation, XGNN is mainly evaluated at an NVLink platform with
- 8 * NVIDIA V100 GPUs (16GB of memory each, SXM2)
- One Intel Xeon Platinum 8163 CPU (total 32 cores),
- 256GB RAM


## Installation

### Software Version

- Ubuntu 18.04 or Ubuntu 20.04
- CMake >= 3.14
- CUDA v11.7
- NVIDIA GPU Driver v525
- Python v3.8
- PyTorch v1.10
- DGL V0.9.1

### Install CUDA11.7

XGNN is built on CUDA 11.7. Follow the instructions in https://developer.nvidia.com/cuda-11-7-0-download-archive to install CUDA 11.7, and make sure that `/usr/local/cuda` is linked to `/usr/local/cuda-11.7`.

### Install GNN Training Systems

We use conda to manage our python environment.

1. We use conda to manage our python environment.

    ```bash
    conda create -n samgraph_env python==3.8 pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge -y # install pytorch 1.10
    conda activate samgraph_env
    conda install cudnn numpy scipy networkx tqdm pandas ninja cmake -y # System cmake is too old to build DGL
    sudo apt install gnuplot # Install gnuplot for experiments:
    ```


2. Download GNN systems.

    ```bash
    # Download XGNN source code
    git clone --recursive https://github.com/lixiaobai09/xgnn.git
    ```

3. Install DGL and FastGraph. The package FastGraph is used to load datasets for GNN systems in all experiments.

    ```bash
    # Install DGL
    ./xgnn/3rdparty/dgl_install.sh

    # Install fastgraph
    ./xgnn/utility/fg_install.sh
    ```

    

4. Install XGNN (also called SamGraph).
   
    ```bash
    cd xgnn
    ./build.sh
    ```

### Change ULIMIT and Open Transparent Huge Pages
Both DGL and XGNN need to use a lot of system resources. DGL sampling requires cro-processing communications while XGNN's data storage in host memory requires memlock(pin) memory to enable faster access between host memory and GPU memory.
To speed up the dataset loading, we recommend open the Transparent Huge Pages (THP) in the Linux system.
Hence we have to set the user limit and open THP.


Append the following content to `/etc/security/limits.conf` and then `reboot`:

```bash
* soft nofile unlimited
* hard nofile unlimited
* soft memlock unlimited
* hard memlock unlimited
```

After reboot you can see:

```bash
> ulimit -n
unlimited

> ulimit -l
unlimited
```

Run the following commands as root privilege to open THP.
```bash
# for normal memory type
echo "always" > /sys/kernel/mm/transparent_hugepage/enabled
# for shared memory
echo "always" > /sys/kernel/mm/transparent_hugepage/shmem_enabled
```

### Docker Support
We provide a Dockerfile to build the experiment image. The file is in the root directory of this repository. Users can use the following command to create a Docker environment.

```bash
docker build . -t xgnn:1.0 -f ./Dockerfile
```

Then users can run tests in Docker.
```bash
docker run --ulimit memlock=-1 --shm-size 256G --rm --gpus all -v ${HOST_DATA_DIR}:/graph-learning -it xgnn:1.0 bash
```

We also support the docker images for the quiver and wholegraph systems. Use the following commands to run these systems. and the source code of these systems can be found in the docker.
```bash
# run quiver with docker image
# DATASET_PATH: the xgnn dataset path
# LOG_PATH: the directory to store quiver logs
# DATASET_CACHE_ROOT: the directory to store temporary dataset in quiver
docker run --name "quiver_eval" \
    --mount type=bind,source=${DATASET_PATH},target=/graph-learning,readonly \
    --mount type=bind,source=${LOG_PATH},target=/logs \
    --mount type=bind,source=${DATASET_CACHE_ROOT},target=/quiver-baseline \
    --env APP_RREFIX=/quiver/benchmarks \
    --rm -it --gpus=all --shm-size=192g anlarry/quiver /bin/bash /quiver/eval_entry.sh

# run wholegraph with docker image
# DATASET_PATH: the xgnn dataset path
# LOG_PATH: the directory to store wholegraph logs
docker run --name "wholegraph_eval" \
    --mount type=bind,source=${DATASET_PATH},target=/graph-learning,readonly \
    --mount type=bind,source=${LOG_PATH},target=/logs \
    --env APP_RREFIX=/wholegraph/examples/gnn_v2 \
    --rm -it --gpus=all --ipc=host anlarry/wholegraph  /bin/bash /wholegraph/examples/gnn_v2/run_8xA100.sh
```

## Dataset Preprocessing

See [`datagen/README.md`](datagen/README.md) to find out how to preprocess datasets.

## QuickStart: Use XGNN to train GNN models

XGNN is compiled into Python library. We have written several GNN models using XGNN’s APIs. These models are in `xgnn/example/samgraph/sgnn` and are easy to run as following:

```bash
cd xgnn/example

python samgraph/sgnn/train_gcn.py --num-worker 4 --cache-policy degree --sample-type khop3 --batch-size 6000 --num-epoch 10 --dataset papers100M --part-cache --gpu-extract --use-dist-graph 1.0 --cache-percentage 0.64
```

## Experiments

Our experiments have been automated by scripts (`run.sh`). Each figure or table in our paper is treated as one experiment and is associated with a subdirectory in `xgnn/evaluation`.



## License

XGNN is released under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0.html).

