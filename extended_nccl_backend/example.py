import os
from typing import cast

import torch
import torch.distributed as dist

import ext_nccl_backend
from ext_nccl_backend import ExtProcessGroupNCCL
from src import nvtx

# init process group
dist.init_process_group(
    backend="cpu:gloo,cuda:ext_nccl_backend",
    # backend="ext_nccl_backend", # NOTE: ext_nccl_backend is not supported for cpu
)

# get rank, world_size and init device
rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
device = torch.cuda.current_device()

# just print the function name to see if it is loaded
print(f"[RANK {rank}] {ext_nccl_backend.createExtProcessGroupNCCL=}")

# get the process group backend
world_group = dist.group.WORLD
print(f"[RANK {rank}] WorldGroup: {type(world_group)=}", f"{world_group._get_backend_name()=}")
backend: dist.Backend = world_group._get_backend(torch.device(device))
print(f"[RANK {rank}] WorldGroup: {type(backend)=}")
assert isinstance(backend, ExtProcessGroupNCCL), (
    f"expected ExtProcessGroupNCCL, got {type(backend)=}"
)

# pg = dist.new_group(list(range(world_size)), backend="ext_nccl_backend")
# print(f"[RANK {rank}] {type(pg)=}", f"{pg._get_backend_name()=}")
# backend: dist.Backend = pg._get_backend(torch.device(device))
# print(f"[RANK {rank}] NewGroup: {type(backend)=}")
# assert isinstance(backend, ExtProcessGroupNCCL), (
#     f"expected ExtProcessGroupNCCL, got {type(backend)=}"
# )

backend: ExtProcessGroupNCCL = cast(ExtProcessGroupNCCL, backend)

torch.cuda.profiler.start()

# BUG: we cannot fetch the nccl stream at this point
# since both the nccl stream and nccl comm are lazily initialized
# until the first collective call
# print(f"[RANK {rank}] {backend.nccl_stream=}")

side_stream = torch.cuda.Stream()
print(f"[RANK {rank}] {side_stream=}")

# --- init data --- #

x = torch.zeros(world_size) + rank
y = x.to(device)
z = y.clone()

m,n,k = 8192, 8192, 1024
a = torch.randn(m, k, device=device)
b = torch.randn(k, n, device=device)
s = torch.randn((m,k), device=device, dtype=torch.float32)
g = torch.empty((m*world_size, k), device=device, dtype=torch.float32)


# this goes through gloo backend
dist.all_reduce(x, group=world_group)
ans = world_size * (world_size - 1) // 2
print(f"[RANK {rank}] expected all-reduce value: {ans=}, and actual value: {x=}") # the result should be [ans] * size

# these go through extended nccl backend
with nvtx.add_nvtx_event("nccl_stream allgather"):
    work = dist.all_gather_into_tensor(
        g,
        s,
        group=world_group,
        async_op=True
    )

# dist.all_reduce(y, group=world_group)  # the result should be [ans] * size
# print(f"[RANK {rank}] cuda allreduce: {y}")

# dist.broadcast(z, 0, group=pg) # the result should be [0] * size
# print(f"[RANK {rank}] cuda broadcast: {z}")

work.wait()
# print(f"[RANK {rank}] all_gather {s=} into tensor: {g=}")

nccl_stream = backend.nccl_stream
print(f"[RANK {rank}] {nccl_stream=}")

with nvtx.add_nvtx_event("side_stream matmul"):
    side_stream.wait_stream(torch.cuda.default_stream())
    with torch.cuda.stream(side_stream):
        c = a @ b

# FIXME: it seems like this nccl stream is valid
# but not the same stream the nccl comm kernel runs on
with nvtx.add_nvtx_event("nccl_stream matmul"):
    nccl_stream.wait_stream(torch.cuda.default_stream())
    with torch.cuda.stream(nccl_stream):
            d = a @ b

dist.barrier()
torch.cuda.synchronize()
torch.cuda.profiler.stop()
dist.destroy_process_group()