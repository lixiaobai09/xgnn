# Figure16: Performance on Other Platforms

The goal of this experiment is to compare the performance between quiver, DGL+C(SGNN) and XGNN on other platforms.

There are two hardware platforms for this experiment:
- 2xPCIe platform:
    - CPU: 2 x Intel(R) Xeon(R) CPU E5-2650 v4 (total 48 cores)
    - GPU: 2 x NVIDIA Tesla V100 (32GB) with PCIe interconnects
    - RAM: 256GB
- 4xNVLink platform:
    - CPU: 2 x Intel Xeon Gold 6138 CPUs (total 80 cores)
    - GPU: 4 x full-NVLink-connected NVIDIA Tesla V100 (16GB memory, SXM2)
    - RAM: 378GB

To run the script file "run_2xPCIe.sh" for the 2xPCIe platform,
and "run_4xNVLink.sh" for the 4xNVLink platform.