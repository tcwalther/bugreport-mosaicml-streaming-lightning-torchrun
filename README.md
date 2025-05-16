# Bug report: MosaicML Streaming vs Torchrun

This repo reproduces a bug in MosaicML Streaming (May 2025) when used on multi-GPU training with torchrun. The
program is stuck in NCCL communication errors.

```
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (24/34) after sleep for 2400 msec
...
```

This doesn't happen with torchdata.

This repo reproduces the bug. It compares a torch.data Dataset and a MosaicML streaming dataset, showing that the former works fine while the latter hangs. To simplify the multinode training code, we
use Pytorch Lightning.

## Setup

Run `uv sync`, or if you don't use `uv`, install `lightning`, `mosaicml-streaming` and `torchvision` (the latter for the MNIST dataset we use as an example).

## Single node

On single node, everything works fine:

```bash
uv run train_mnist_torchdata.py  # this works fine
```

```bash
uv run convert_mnist_to_streaming.py
uv run train_mnist_streaming.py  # this also works fine
```

## Multi node

Then proceed to multi-process training. You need a multi-GPU NVidia instance to use NCCL. You will see that the torchdata case trains fine, but the streaming case is stuck. Setting `NCCL_DEBUG=INFO` will show underlying the communication problem.

```bash
$ NCCL_DEBUG=INFO uv run torchrun --nproc_per_node=2 ./train_mnist_torchdata.py  # this works fine
```

```bash
$ NCCL_DEBUG=INFO uv run torchrun --nproc_per_node=2 ./train_mnist_streaming.py  # This is stuck
W0516 10:48:17.790000 10143 torch/distributed/run.py:766]
W0516 10:48:17.790000 10143 torch/distributed/run.py:766] *****************************************
W0516 10:48:17.790000 10143 torch/distributed/run.py:766] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
W0516 10:48:17.790000 10143 torch/distributed/run.py:766] *****************************************
[rank0]:[W516 10:48:21.037728976 ProcessGroupNCCL.cpp:4715] [PG ID 0 PG GUID 0 Rank 0]  using GPU 0 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.
[rank1]:[W516 10:48:21.038515993 ProcessGroupNCCL.cpp:4715] [PG ID 0 PG GUID 0 Rank 1]  using GPU 1 as device used by this process is currently unknown. This can potentially cause a hang if this rank to GPU mapping is incorrect. You can pecify device_id in init_process_group() to force use of a particular device.
vae-bc5a469e-2296-0:10208:10208 [0] NCCL INFO Bootstrap: Using eth0:10.105.0.5<0>
vae-bc5a469e-2296-0:10208:10208 [0] NCCL INFO cudaDriverVersion 12020
vae-bc5a469e-2296-0:10208:10208 [0] NCCL INFO NCCL version 2.26.2+cuda12.2
vae-bc5a469e-2296-0:10208:10208 [0] NCCL INFO Comm config Blocking set to 1
vae-bc5a469e-2296-0:10209:10209 [1] NCCL INFO cudaDriverVersion 12020
vae-bc5a469e-2296-0:10209:10209 [1] NCCL INFO Bootstrap: Using eth0:10.105.0.5<0>
vae-bc5a469e-2296-0:10209:10209 [1] NCCL INFO NCCL version 2.26.2+cuda12.2
vae-bc5a469e-2296-0:10209:10209 [1] NCCL INFO Comm config Blocking set to 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. Using internal net plugin.
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Failed to open libibverbs.so[.1]
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO NET/Socket : Using [0]eth0:10.105.0.5<0>
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so.
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Using network Socket
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO ncclCommInitRankConfig comm 0x58892a1d3910 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 100000 commId 0x56bea185a98d5193 - Init START
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO NET/Plugin: Could not find: libnccl-net.so. Using internal net plugin.
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO Failed to open libibverbs.so[.1]
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO NET/Socket : Using [0]eth0:10.105.0.5<0>
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so.
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO Using network Socket
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO ncclCommInitRankConfig comm 0x650f74e56d60 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 200000 commId 0x56bea185a98d5193 - Init START
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO RAS client listening socket at 127.0.0.1<28028>
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO Bootstrap timings total 0.036238 (create 0.000045, send 0.030891, recv 0.000203, ring 0.000082, delay 0.000000)
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Bootstrap timings total 0.218519 (create 0.000039, send 0.000110, recv 0.217947, ring 0.000033, delay 0.000000)
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO Setting affinity for GPU 1 to ffff,ffffff00,00000000
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Setting affinity for GPU 0 to ff,ffffffff
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO comm 0x650f74e56d60 rank 1 nRanks 2 nNodes 1 localRanks 2 localRank 1 MNNVL 0
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO comm 0x58892a1d3910 rank 0 nRanks 2 nNodes 1 localRanks 2 localRank 0 MNNVL 0
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO Trees [0] -1/-1/-1->1->0 [1] -1/-1/-1->1->0 [2] -1/-1/-1->1->0 [3] -1/-1/-1->1->0 [4] 0/-1/-1->1->-1 [5] 0/-1/-1->1->-1 [6] 0/-1/-1->1->-1 [7] 0/-1/-1->1->-1 [8] -1/-1/-1->1->0 [9] -1/-1/-1->1->0 [10] -1/-1/-1->1->0 [11] -1/-1/-1->1->0 [12] 0/-1/-1->1->-1 [13] 0/-1/-1->1->-1 [14] 0/-1/-1->1->-1 [15] 0/-1/-1->1->-1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 00/16 : 0 1
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO P2P Chunksize set to 524288
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 01/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 02/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 03/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 04/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 05/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 06/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 07/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 08/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 09/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 10/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 11/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 12/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 13/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 14/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Channel 15/16 : 0 1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1 [2] 1/-1/-1->0->-1 [3] 1/-1/-1->0->-1 [4] -1/-1/-1->0->1 [5] -1/-1/-1->0->1 [6] -1/-1/-1->0->1 [7] -1/-1/-1->0->1 [8] 1/-1/-1->0->-1 [9] 1/-1/-1->0->-1 [10] 1/-1/-1->0->-1 [11] 1/-1/-1->0->-1 [12] -1/-1/-1->0->1 [13] -1/-1/-1->0->1 [14] -1/-1/-1->0->1 [15] -1/-1/-1->0->1
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO P2P Chunksize set to 524288
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Check P2P Type intraNodeP2pSupport 1 directMode 0
vae-bc5a469e-2296-0:10209:10230 [1] NCCL INFO [Proxy Service] Device 1 CPU core 56
vae-bc5a469e-2296-0:10208:10231 [0] NCCL INFO [Proxy Service] Device 0 CPU core 30
vae-bc5a469e-2296-0:10209:10232 [1] NCCL INFO [Proxy Service UDS] Device 1 CPU core 64
vae-bc5a469e-2296-0:10208:10233 [0] NCCL INFO [Proxy Service UDS] Device 0 CPU core 33
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO 16 coll channels, 16 collnet channels, 0 nvls channels, 16 p2p channels, 16 p2p channels per peer
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO 16 coll channels, 16 collnet channels, 0 nvls channels, 16 p2p channels, 16 p2p channels per peer
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO CC Off, workFifoBytes 1048576
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO ncclCommInitRankConfig comm 0x58892a1d3910 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 100000 commId 0x56bea185a98d5193 - Init COMPLETE
vae-bc5a469e-2296-0:10208:10226 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 2 total 0.61 (kernels 0.19, alloc 0.08, bootstrap 0.22, allgathers 0.00, topo 0.12, graphs 0.00, connections 0.01, rest 0.00)
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO TUNER/Plugin: Could not find: libnccl-tuner.so. Using internal tuner plugin.
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO ncclCommInitRankConfig comm 0x650f74e56d60 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 200000 commId 0x56bea185a98d5193 - Init COMPLETE
vae-bc5a469e-2296-0:10209:10227 [1] NCCL INFO Init timings - ncclCommInitRankConfig: rank 1 nranks 2 total 0.48 (kernels 0.23, alloc 0.08, bootstrap 0.04, allgathers 0.00, topo 0.12, graphs 0.00, connections 0.01, rest 0.00)
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 02/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 03/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 00/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 04/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 01/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 05/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 02/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 06/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 03/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 07/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 04/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 08/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 05/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 09/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 06/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 10/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 07/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 11/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 08/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 12/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 09/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 13/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 10/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 14/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 11/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Channel 15/0 : 0[0] -> 1[1] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 12/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 13/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 14/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Channel 15/0 : 1[1] -> 0[0] via P2P/CUMEM
vae-bc5a469e-2296-0:10209:10235 [1] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
vae-bc5a469e-2296-0:10208:10234 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
vae-bc5a469e-2296-0:10209:10239 [1] NCCL INFO comm 0x650f74e56d60 rank 1 nranks 2 cudaDev 1 busId 200000 - Destroy COMPLETE
Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/2
vae-bc5a469e-2296-0:10209:10209 [1] NCCL INFO Comm config Blocking set to 1
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO Using network Socket
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO ncclCommInitRankConfig comm 0x650f7526ddc0 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 200000 commId 0x56bea185a98d5193 - Init START
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (1/34) after sleep for 100 msec
vae-bc5a469e-2296-0:10208:10240 [0] NCCL INFO comm 0x58892a1d3910 rank 0 nranks 2 cudaDev 0 busId 100000 - Destroy COMPLETE
Using default `ModelCheckpoint`. Consider installing `litmodels` package to enable `LitModelCheckpoint` for automatic upload to the Lightning model registry.
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/connectors/logger_connector/logger_connector.py:76: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `lightning.pytorch` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
You are using a CUDA device ('NVIDIA H100 NVL') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All distributed processes registered. Starting with 2 processes
----------------------------------------------------------------------------------------------------

vae-bc5a469e-2296-0:10208:10208 [0] NCCL INFO Comm config Blocking set to 1
vae-bc5a469e-2296-0:10208:10247 [0] NCCL INFO Using network Socket
vae-bc5a469e-2296-0:10208:10247 [0] NCCL INFO ncclCommInitRankConfig comm 0x58892a5e7910 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 100000 commId 0x139a27671bffc886 - Init START
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (2/34) after sleep for 200 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (3/34) after sleep for 300 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (4/34) after sleep for 400 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (5/34) after sleep for 500 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (6/34) after sleep for 600 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (7/34) after sleep for 700 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (8/34) after sleep for 800 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (9/34) after sleep for 900 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (10/34) after sleep for 1000 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (11/34) after sleep for 1100 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (12/34) after sleep for 1200 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (13/34) after sleep for 1300 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (14/34) after sleep for 1400 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (15/34) after sleep for 1500 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (16/34) after sleep for 1600 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (17/34) after sleep for 1700 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (18/34) after sleep for 1800 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (19/34) after sleep for 1900 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (20/34) after sleep for 2000 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (21/34) after sleep for 2100 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (22/34) after sleep for 2200 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (23/34) after sleep for 2300 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (24/34) after sleep for 2400 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (25/34) after sleep for 2500 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (26/34) after sleep for 2600 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (27/34) after sleep for 2700 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (28/34) after sleep for 2800 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (29/34) after sleep for 2900 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (30/34) after sleep for 3000 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (31/34) after sleep for 3100 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (32/34) after sleep for 3200 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (33/34) after sleep for 3300 msec
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (34/34) after sleep for 3400 msec

[2025-05-16 10:49:23] vae-bc5a469e-2296-0:10209:10243 [1] misc/socket.cc:544 NCCL WARN socketPollConnect: connect returned Connection refused, exceeded error retry count (35)
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO misc/socket.cc:635 -> 6
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO misc/socket.cc:684 -> 6
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO bootstrap.cc:605 -> 6
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO bootstrap.cc:687 -> 6
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO init.cc:1404 -> 6
vae-bc5a469e-2296-0:10209:10243 [1] NCCL INFO group.cc:75 -> 6 [Async thread]
vae-bc5a469e-2296-0:10209:10209 [1] NCCL INFO group.cc:422 -> 6
vae-bc5a469e-2296-0:10209:10209 [1] NCCL INFO group.cc:581 -> 6
vae-bc5a469e-2296-0:10209:10209 [1] NCCL INFO init.cc:1836 -> 6
[rank1]: [rank1]: Traceback (most recent call last):
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/./train_mnist_streaming.py", line 79, in <module>
[rank1]: [rank1]:     main()
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/./train_mnist_streaming.py", line 75, in main
[rank1]: [rank1]:     trainer.fit(model=model, train_dataloaders=dataloader)
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 561, in fit
[rank1]: [rank1]:     call._call_and_handle_interrupt(
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 47, in _call_and_handle_interrupt
[rank1]: [rank1]:     return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
[rank1]: [rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/strategies/launchers/subprocess_script.py", line 105, in launch
[rank1]: [rank1]:     return function(*args, **kwargs)
[rank1]: [rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 599, in _fit_impl
[rank1]: [rank1]:     self._run(model, ckpt_path=ckpt_path)
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 969, in _run
[rank1]: [rank1]:     self.__setup_profiler()
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1102, in __setup_profiler
[rank1]: [rank1]:     self.profiler.setup(stage=self.state.fn, local_rank=local_rank, log_dir=self.log_dir)
[rank1]: [rank1]:                                                                             ^^^^^^^^^^^^
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1264, in log_dir
[rank1]: [rank1]:     dirpath = self.strategy.broadcast(dirpath)
[rank1]: [rank1]:               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/strategies/ddp.py", line 307, in broadcast
[rank1]: [rank1]:     torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
[rank1]: [rank1]:     return func(*args, **kwargs)
[rank1]: [rank1]:            ^^^^^^^^^^^^^^^^^^^^^
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 3483, in broadcast_object_list
[rank1]: [rank1]:     broadcast(object_sizes_tensor, src=global_src, group=group)
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
[rank1]: [rank1]:     return func(*args, **kwargs)
[rank1]: [rank1]:            ^^^^^^^^^^^^^^^^^^^^^
[rank1]: [rank1]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 2714, in broadcast
[rank1]: [rank1]:     work = group.broadcast([tensor], opts)
[rank1]: [rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank1]: [rank1]: torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/NCCLUtils.cpp:77, remote process exited or there was a network error, NCCL version 2.26.2
[rank1]: [rank1]: ncclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.
[rank1]: [rank1]: Last error:
[rank1]: [rank1]: socketPollConnect: connect returned Connection refused, exceeded error retry count (35)
W0516 10:49:23.951000 10143 torch/distributed/elastic/multiprocessing/api.py:900] Sending process 10208 closing signal SIGTERM
/home/azureuser/.local/share/uv/python/cpython-3.11.12-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 6 leaked shared_memory objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
E0516 10:49:24.165000 10143 torch/distributed/elastic/multiprocessing/api.py:874] failed (exitcode: 1) local_rank: 1 (pid: 10209) of binary: /home/azureuser/sky_workdir/.venv/bin/python
Traceback (most recent call last):
  File "/home/azureuser/sky_workdir/.venv/bin/torchrun", line 10, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 355, in wrapper
    return f(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/run.py", line 892, in main
    run(args)
  File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/run.py", line 883, in run
    elastic_launch(
  File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 139, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 270, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
./train_mnist_streaming.py FAILED
------------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-05-16_10:49:23
  host      : vae-bc5a469e-2296-0.internal.cloudapp.net
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 10209)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
```
