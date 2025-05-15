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


```
$ sky exec -c vae sky_streaming.yaml

...

(worker1, rank=1, pid=27522, ip=10.105.0.4) vae-bc5a469e-2296-2:27720:27757 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (33/34) after sleep for 3300 msec
(worker3, rank=3, pid=29549, ip=10.105.0.7) vae-bc5a469e-2296-3:29744:29780 [0] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (34/34) after sleep for 3400 msec
(worker1, rank=1, pid=27522, ip=10.105.0.4) vae-bc5a469e-2296-2:27719:27754 [0] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (34/34) after sleep for 3400 msec
(worker1, rank=1, pid=27522, ip=10.105.0.4) vae-bc5a469e-2296-2:27720:27757 [1] NCCL INFO socketPollConnect: connect returned Connection refused, retrying (34/34) after sleep for 3400 msec

...

(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]: Traceback (most recent call last):
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/./train_mnist_streaming.py", line 79, in <module>
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     main()
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/./train_mnist_streaming.py", line 75, in main
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     trainer.fit(model=model, train_dataloaders=dataloader)
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 561, in fit
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     call._call_and_handle_interrupt(
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/call.py", line 47, in _call_and_handle_interrupt
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/strategies/launchers/subprocess_script.py", line 105, in launch
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     return function(*args, **kwargs)
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:            ^^^^^^^^^^^^^^^^^^^^^^^^^
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 599, in _fit_impl
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     self._run(model, ckpt_path=ckpt_path)
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 969, in _run
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     self.__setup_profiler()
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1102, in __setup_profiler
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     self.profiler.setup(stage=self.state.fn, local_rank=local_rank, log_dir=self.log_dir)
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:                                                                             ^^^^^^^^^^^^
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/trainer/trainer.py", line 1264, in log_dir
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     dirpath = self.strategy.broadcast(dirpath)
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/lightning/pytorch/strategies/ddp.py", line 307, in broadcast
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     return func(*args, **kwargs)
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:            ^^^^^^^^^^^^^^^^^^^^^
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 3483, in broadcast_object_list
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     broadcast(object_sizes_tensor, src=global_src, group=group)
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     return func(*args, **kwargs)
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:            ^^^^^^^^^^^^^^^^^^^^^
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:   File "/home/azureuser/sky_workdir/.venv/lib/python3.11/site-packages/torch/distributed/distributed_c10d.py", line 2714, in broadcast
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:     work = group.broadcast([tensor], opts)
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]: torch.distributed.DistBackendError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/NCCLUtils.cpp:77, remote process exited or there was a network error, NCCL version 2.26.2
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]: ncclRemoteError: A call failed possibly due to a network error or a remote process exiting prematurely.
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]: Last error:
(worker1, rank=1, pid=27522, ip=10.105.0.4) [rank4]: [rank4]: socketPollConnect: connect returned Connection refused, exceeded error retry count (35)

...
```