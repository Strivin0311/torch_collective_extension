import os

import torch
import torch.distributed as dist

import ext_nccl_backend

dist.init_process_group("cpu:gloo,cuda:ext_nccl_backend")

rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)

ans = world_size * (world_size - 1) // 2
print(f"[RANK {rank}] expected all-reduce value: {ans=}")

size = 5

x = torch.zeros(size) + rank
y = x.to(torch.cuda.current_device())
z = y.clone()

# this goes through gloo backend
dist.all_reduce(x)
print(f"[RANK {rank}] cpu allreduce: {x}") # the result should be [ans] * size

# this goes through extended nccl backend
dist.all_reduce(y)  # the result should be [ans] * size
print(f"[RANK {rank}] cuda allreduce: {y}")

dist.broadcast(z, 0) # the result should be [0] * size
print(f"[RANK {rank}] cuda broadcast: {z}")


dist.barrier()
dist.destroy_process_group()