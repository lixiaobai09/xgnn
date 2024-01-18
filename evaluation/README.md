# Experiments

### Table of Contents 
  - [Overview](#overview)
  - [Paper's Hardware Configurations](#papers-hardware-configurations)
  - [Run A Single Experiment](#run-a-single-experiment)

## Overview
Our experiments have been automated by scripts (`run.sh`).
Each figure or table in our paper is treated as one experiment and is associated with a subdirectory in `xgnn/evaluation`.
The script will automatically run the experiment, save the logs into files, and parse the output data from the files.

```bash
> tree -L 2 evaluation
evaluation
├── README.md
├── figure10
│   ├── figure.plt
│   ├── parse_res.py
│   └── run.sh
├── figure11
│   ├── figure.plt
│   ├── parse_res.py
│   └── run.sh
├── ... 
├── gnnlab_cmp
│   └── run.sh
├── large_graph
│   └── run.sh
├── other_platform
│   ├── run.sh
│   └── run_pcie.sh
...
```
## Paper's Hardware Configurations
- 8 * NVIDIA V100 GPUs (16GB of memory each, SXM2)
- One Intel Xeon Platinum 8163 CPU (total 32 cores),
- 256GB RAM

**Note: If you have a different hardware environment, you need to goto the subdirectories (i.e., `figXX` or `tableXX`), follow the instructions to modify some script configurations(e.g. smaller cache ratio), and then run the experiment**


## Run A Single Experiment

The following commands are used to run a certain experiment(e.g. figure10).

```bash
cd xgnn/evaluation/figure10
bash run.sh
```