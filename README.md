# Bug report: MosaicML Streaming vs Torchrun

This repo reproduces a bug in MosaicML Streaming (May 2025) when used on multi-node training with torchrun. The
program is stuck in NCCL communication errors.

```
(worker3, rank=3, pid=16415, ip=10.105.0.7) test-bc5a469e-2296-3:16728:16793 [0] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (30/34) after sleep for 3000 msec
(worker1, rank=1, pid=18446, ip=10.105.0.4) test-bc5a469e-2296-2:18759:18839 [0] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (30/34) after sleep for 3000 msec
(worker2, rank=2, pid=18045, ip=10.105.0.6) test-bc5a469e-2296-1:18357:18434 [0] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (30/34) after sleep for 3000 msec
(worker3, rank=3, pid=16415, ip=10.105.0.7) test-bc5a469e-2296-3:16728:16793 [0] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (31/34) after sleep for 3100 msec
...
```

This doesn't happen with other datasets.

This repo reproduces the bug. It relies on torch and streaming. The issue also occurs on torch 2.7 which ships with a newer version of NCCL. To reproduce, start with single node training:

```bash
uv run train_mnist_torchdata.py  # this works fine
```

```bash
uv run convert_mnist_to_streaming.py
uv run train_mnist_streaming.py  # this also works fine
```

Then proceed to multi-node training. This repo provides Skypilot files, `sky_torchdata.yaml` and `sky_streaming.yaml` in case you're using Skypilot to provision
your machines. Otherwise, just copy the run steps from these to files. You will see that the torchdata case trains fine, but the streaming case is stuck.