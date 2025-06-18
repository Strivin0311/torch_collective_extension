import os

import torch
import torch.distributed as dist

import dummy_collectives_backend

dist.init_process_group("cpu:gloo,cuda:dummy_backend")

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
        print("got RuntimeError when calling broadcast since dummy backend does not support it")