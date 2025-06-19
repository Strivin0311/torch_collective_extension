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
backend: ExtProcessGroupNCCL = cast(ExtProcessGroupNCCL, backend)

pg = dist.new_group(list(range(world_size)), backend="ext_nccl_backend")
print(f"[RANK {rank}] {type(pg)=}", f"{pg._get_backend_name()=}")
pg_backend: dist.Backend = pg._get_backend(torch.device(device))
print(f"[RANK {rank}] NewGroup: {type(pg_backend)=}")
assert isinstance(pg_backend, ExtProcessGroupNCCL), (
    f"expected ExtProcessGroupNCCL, got {type(pg_backend)=}"
)

# --- init data --- #

x = torch.zeros(world_size) + rank
y = x.to(device)
z = y.clone()


# --- try simple functionalities --- #

# NOTE: we cannot fetch the nccl stream at this point
# since both the nccl stream and nccl comm are lazily initialized
# until the first collective call
# print(f"[RANK {rank}] {backend.nccl_stream=}")

# this goes through gloo backend
dist.all_reduce(x, group=world_group)
ans = world_size * (world_size - 1) // 2
print(f"[RANK {rank}] expected all-reduce value: {ans=}, and actual value: {x=}") # the result should be [ans] * size

# this goes through nccl backend
dist.all_reduce(y, group=world_group)  # the result should be [ans] * size
print(f"[RANK {rank}] cuda allreduce: {y}")

dist.broadcast(z, 0, group=pg) # the result should be [0] * size
print(f"[RANK {rank}] cuda broadcast: {z}")


# --- try multi-stream --- #

side_stream = torch.cuda.Stream()
print(f"[RANK {rank}] {side_stream=} | {side_stream.stream_id=} | {side_stream.device_index=} | {side_stream.device_type=}")

nccl_stream = backend.nccl_stream
print(f"[RANK {rank}] {nccl_stream=} | {nccl_stream.stream_id=} | {nccl_stream.device_index=} | {nccl_stream.device_type=}")

m,n,k = 16384, 16384, 8192
a = torch.randn(m, k, device=device)
b = torch.randn(k, n, device=device)
s = torch.randn((m,n), device=device, dtype=torch.float32)
g = torch.empty((m*world_size, n), device=device, dtype=torch.float32)

profile_mode = os.environ.get("EXAMPLE_PROFILE_MODE", "0") == "1"
prof_iters, prof_start_iter, prof_end_iter = 10, 5, 8

for iter in range(prof_iters):
    if profile_mode:
        nvtx.switch_profile(
            iter,
            prof_start_iter,
            prof_end_iter,
            profile_ranks=[0],
        )
    
    # refetch the nccl stream to test the consistency
    nccl_stream = backend.nccl_stream
    print(f"[RANK {rank}] iter {iter} {nccl_stream=} | {nccl_stream.stream_id=} | {nccl_stream.device_index=} | {nccl_stream.device_type=}")
    
    with nvtx.add_nvtx_event("nccl_stream allgather"):
        work = dist.all_gather_into_tensor(
            g,
            s,
            group=world_group,
            async_op=True
        )
        
    side_stream.wait_stream(torch.cuda.default_stream())
    with nvtx.add_nvtx_event("side_stream matmul"):
        with torch.cuda.stream(side_stream):
            c = a @ b
    
    nccl_stream.wait_stream(torch.cuda.default_stream())
    with nvtx.add_nvtx_event("nccl_stream matmul"):
        with torch.cuda.stream(nccl_stream):
                d = a @ b

    work.wait()
    with nvtx.add_nvtx_event("default_stream matmul"):
        e = a @ b
        
    dist.barrier()
    torch.cuda.synchronize()
        
        
dist.barrier()
torch.cuda.synchronize()
dist.destroy_process_group()