import os

import torch
import torch.distributed as dist

import dummy_collectives_pg

dist.init_process_group("dummy_pg")

x = torch.ones(6)
dist.all_reduce(x)
print(f"cpu allreduce: {x}")

if torch.cuda.is_available():
    y = x.cuda()
    dist.all_reduce(y)
    print(f"cuda allreduce: {y}")

try:
    dist.broadcast(x, 0)
except RuntimeError:
    print("got RuntimeError when calling broadcast since dummy process group does not support it")
