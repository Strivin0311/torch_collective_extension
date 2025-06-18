import os
from typing import cast

import torch
import torch.distributed as dist

import ext_nccl_backend
from ext_nccl_backend import ExtProcessGroupNCCL

dist.init_process_group(
    backend="cpu:gloo,cuda:ext_nccl_backend",
    # backend="ext_nccl_backend", # NOTE: ext_nccl_backend is not supported for cpu
)

rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
device = torch.cuda.current_device()

# just print the function name to see if it is loaded
print(f"[RANK {rank}] {ext_nccl_backend.createExtProcessGroupNCCL=}")

world_group = dist.group.WORLD
print(f"[RANK {rank}] WorldGroup: {type(world_group)=}", f"{world_group._get_backend_name()=}")
backend: dist.Backend = world_group._get_backend(torch.device(device))
print(f"[RANK {rank}] WorldGroup: {type(backend)=}")
# backend = cast(ExtProcessGroupNCCL, backend)
assert isinstance(backend, ExtProcessGroupNCCL), (
    f"expected ExtProcessGroupNCCL, got {type(backend)=}"
)

pg = dist.new_group(list(range(world_size)), backend="ext_nccl_backend")
print(f"[RANK {rank}] {type(pg)=}", f"{pg._get_backend_name()=}")
backend: dist.Backend = pg._get_backend(torch.device(device))
print(f"[RANK {rank}] NewGroup: {type(backend)=}")
# backend = cast(ExtProcessGroupNCCL, backend)
assert isinstance(backend, ExtProcessGroupNCCL), (
    f"expected ExtProcessGroupNCCL, got {type(backend)=}"
)

ans = world_size * (world_size - 1) // 2
print(f"[RANK {rank}] expected all-reduce value: {ans=}")

size = 5

x = torch.zeros(size) + rank
y = x.to(device)
z = y.clone()

# this goes through gloo backend
dist.all_reduce(x, group=world_group)
print(f"[RANK {rank}] cpu allreduce: {x}") # the result should be [ans] * size

# this goes through extended nccl backend
dist.all_reduce(y, group=world_group)  # the result should be [ans] * size
print(f"[RANK {rank}] cuda allreduce: {y}")

dist.broadcast(z, 0, group=pg) # the result should be [0] * size
print(f"[RANK {rank}] cuda broadcast: {z}")


dist.barrier()
dist.destroy_process_group()