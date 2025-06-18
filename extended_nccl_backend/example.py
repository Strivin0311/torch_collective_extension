import os

import torch
import torch.distributed as dist

import ext_nccl_backend

dist.init_process_group("cpu:gloo,cuda:ext_nccl_backend")

# this goes through gloo
x = torch.ones(6)
dist.all_reduce(x)
print(f"cpu allreduce: {x}")

# this goes through dummy
if torch.cuda.is_available():
    y = x.cuda()
    dist.all_reduce(y)
    print(f"cuda allreduce: {y}")

    try:
        dist.broadcast(y, 0)
    except RuntimeError:
        print("got RuntimeError when calling broadcast since extended nccl backend does not support it")