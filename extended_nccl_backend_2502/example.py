import os
from typing import cast
from itertools import chain

import torch
import torch.distributed as dist

import ext_nccl_backend
from ext_nccl_backend import ExtProcessGroupNCCL
from src import nvtx
from src.utils import (
    sanity_check_for_group_cast_meta_args_per_rank,
    sanity_check_for_group_reduce_meta_args_per_rank,
)
from src.ext_distributed_c10d import (
    dummy_all_gather_into_tensor,
    extended_all_to_all_single,
    group_cast_collective,
    group_reduce_collective,
)


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
dtype = torch.bfloat16

def print_rank(msg: str):
    """Print the rank and message."""
    rank = int(os.environ["LOCAL_RANK"])
    print(f"\n[RANK {rank}] {msg}\n", flush=True)

# just print the function name to see if it is loaded
print_rank(f"{ext_nccl_backend.createExtProcessGroupNCCL=}")


# --- init pg and backend --- #

# get the process group backend
world_group = dist.group.WORLD
print_rank(f"WorldGroup: {type(world_group)=}, {world_group._get_backend_name()=}")

backend: dist.Backend = world_group._get_backend(torch.device(device))
print_rank(f"WorldGroup: {type(backend)=}")
assert isinstance(backend, ExtProcessGroupNCCL), (
    f"expected ExtProcessGroupNCCL, got {type(backend)=}"
)
backend: ExtProcessGroupNCCL = cast(ExtProcessGroupNCCL, backend)

pg = dist.new_group(list(range(world_size)), backend="ext_nccl_backend")
print_rank(f"NewGroup: {type(pg)=}, {pg._get_backend_name()=}")

pg_backend: dist.Backend = pg._get_backend(torch.device(device))
print_rank(f"NewGroup: {type(pg_backend)=}")
assert isinstance(pg_backend, ExtProcessGroupNCCL), (
    f"expected ExtProcessGroupNCCL, got {type(pg_backend)=}"
)

# --- init data --- #

x = torch.zeros(world_size) + rank
y = x.to(device)
z = y.clone()
p = torch.arange(world_size, device=device, dtype=torch.float32) + rank * 2
gp = torch.empty(world_size**2, device=device, dtype=torch.float32)

q = torch.arange(world_size*2, device=device, dtype=torch.float32) + rank * 2
aq = torch.empty(world_size*2, device=device, dtype=torch.float32)
avq = torch.empty(
    (3 * (world_size // 2)) 
    if rank < world_size - 1 
    else ((world_size + 3) * (world_size // 2)),
    device=device, 
    dtype=torch.float32
)
avq_ext = torch.empty_like(avq)


# prepare for group cast
nh, hd = 2, 3

gc_input_tensor_per_rank = torch.tensor(
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ],
    dtype=dtype,
    device=device,
)
gc_expected_tensor_per_rank = [
    torch.tensor([5, 9, 13], dtype=dtype, device=device),
    torch.tensor([0, 1, 10, 11, 2, 12, 13], dtype=dtype, device=device),
    torch.tensor([2, 3, 6, 7, 14, 15], dtype=dtype, device=device),
    torch.tensor([4, 5, 8, 9], dtype=dtype, device=device),
]
gc_input_tensor = gc_input_tensor_per_rank[rank].repeat_interleave(nh*hd).view(-1, nh, hd)
gc_expected_tensor = gc_expected_tensor_per_rank[rank].repeat_interleave(nh*hd).view(-1, nh, hd)
gc_output_tensor = torch.empty_like(gc_expected_tensor, dtype=dtype, device=device)

gc_input_split_size_list_per_rank = [
    [2, 1, 1], # r0
    [1, 1, 2], # r1
    [1, 1, 2], # r2
    [1, 1, 2], # r3
]
dst_indices_list_per_rank = [
    [[1], [1, 2], [2]], # r0
    [[3], [0, 3], [2]], # r1
    [[3], [0, 3], [1]], # r2
    [[1], [0, 1], [2]], # r3
]
gc_output_split_size_list_per_rank = [
    [1, 1, 1], # r
    [2, 2, 1, 1, 1], # r1 => BUG: [2, 2, 1, 2], # r1
    [1, 1, 2, 2], # r2
    [1, 1, 1, 1], # r3
]
src_index_list_per_rank = [
    [1, 2, 3], # r0
    [0, 2, 0, 3, 3], # r1 => BUG: [0, 2, 0, 3], # r1
    [0, 0, 1, 3], # r2
    [1, 1, 2, 2] # r3
]
sanity_check_for_group_cast_meta_args_per_rank(
    input_split_size_list_per_rank=gc_input_split_size_list_per_rank,
    output_split_size_list_per_rank=gc_output_split_size_list_per_rank,
    dst_indices_list_per_rank=dst_indices_list_per_rank,
    src_index_list_per_rank=src_index_list_per_rank,
    world_size=world_size,
    check_nccl_send_recv=True,
)
gc_input_split_size_list = gc_input_split_size_list_per_rank[rank]
gc_output_split_size_list = gc_output_split_size_list_per_rank[rank]
dst_indices_list = dst_indices_list_per_rank[rank]
src_index_list = src_index_list_per_rank[rank]

# prepare for group reduce
gr_input_tensor_per_rank = [
    torch.tensor([0, 1, 2, 3, 4], dtype=dtype, device=device),
    torch.tensor([5, 6, 7, 8, 9, 10, 11], dtype=dtype, device=device),
    torch.tensor([12, 13, 14, 15, 16], dtype=dtype, device=device),
    torch.tensor([17, 18, 19, 20, 21], dtype=dtype, device=device),
]

gr_output_tensor_per_rank = torch.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
], dtype=dtype, device=device)

gr_expected_tensor_per_rank = torch.tensor([
    [8, 10, 21, 19],
    [17, 18, 13, 14],
    [20, 22, 7, 8],
    [10, 13, 15, 16],
], dtype=dtype, device=device)
gr_input_tensor = gr_input_tensor_per_rank[rank].repeat_interleave(nh*hd).view(-1, nh, hd)
gr_expected_tensor = gr_expected_tensor_per_rank[rank].repeat_interleave(nh*hd).view(-1, nh, hd)
gr_output_tensor = gr_output_tensor_per_rank[rank].repeat_interleave(nh*hd).view(-1, nh, hd)

gr_input_split_size_list_per_rank = [
    [1, 1, 1, 2], # r0
    [2, 2, 1, 1, 1], # r1
    [1, 2, 2], # r2
    [1, 1, 1, 1, 1], # r3 => BUG: [2, 1, 2], # r3
]
dst_index_list_per_rank = [
    [1, 2, 3, 0], # r0
    [0, 2, 0, 3, 3], # r1
    [0, 1, 3], # r2
    [1, 1, 0, 2, 2], # r3 => BUG: [1, 0, 2], # r3
]
gr_output_split_size_list_per_rank = [
    [2, 1, 1], # r0
    [1, 1, 2], # r1
    [1, 1, 2], # r2
    [1, 1, 2], # r3
]
src_indices_list_per_rank = [
    [[0, 1], [1, 2], [3]], # r0
    [[3], [0, 3], [2]], # r1
    [[3], [0, 3], [1]], # r2
    [[1], [0, 1], [2]], # r3
]
sanity_check_for_group_reduce_meta_args_per_rank(
    input_split_size_list_per_rank=gr_input_split_size_list_per_rank,
    output_split_size_list_per_rank=gr_output_split_size_list_per_rank,
    dst_index_list_per_rank=dst_index_list_per_rank,
    src_indices_list_per_rank=src_indices_list_per_rank,
    world_size=world_size,
    check_nccl_send_recv=True,
)
gr_input_split_size_list = gr_input_split_size_list_per_rank[rank]
gr_output_split_size_list = gr_output_split_size_list_per_rank[rank]
dst_index_list = dst_index_list_per_rank[rank]
src_indices_list = src_indices_list_per_rank[rank]


# --- try simple functionalities --- #

# NOTE: we cannot fetch the nccl stream at this point
# since both the nccl stream and nccl comm are lazily initialized
# until the first collective call
# print_rank(f"{backend.nccl_stream=}")

# this goes through gloo backend
dist.all_reduce(x, group=world_group)
ans = world_size * (world_size - 1) // 2
print_rank(f"cpu all-reduce for gloo backend: expected value: {ans=}, and actual value: {x=}") # the result should be [ans] * size

# this goes through nccl backend
# and is expected to the same as nccl all-reduce
dist.all_reduce(y, group=world_group)  # the result should be [ans] * size
print_rank(f"cuda all-reduce for ext_nccl_backend: expected value: {ans=}, and actual value: {y=}")

# this is expected to the same as nccl broadcast
dist.broadcast(z, 0, group=pg) # the result should be [0] * size
print_rank(f"cuda broadcast for ext_nccl_backend: expected value: 0, and actual value: {z=}")

# this is expected to the same as nccl all-gather
work = dist.all_gather_into_tensor(
    output_tensor=gp,
    input_tensor=p,
    group=world_group,
    async_op=True,
)
work.wait()
print_rank(f"cuda all-gather for ext_nccl_backend {p=} into {gp=}")


# this is expected to the same as nccl all-to-all for list of tensors
input = torch.arange(4, device=device, dtype=torch.float32) + rank * 4
input = list(input.chunk(4))
output = list(torch.empty([4], device=device, dtype=torch.float32).chunk(4))
work = dist.all_to_all(output, input, group=world_group, async_op=True)
work.wait()
print_rank(f"cuda all-to-all for ext_nccl_backend {input=} into {output=}")


# this is expected to the same as nccl all-to-all
work = dist.all_to_all_single(
    output=aq,
    input=q,
    output_split_sizes=[2] * world_size,
    input_split_sizes=[2] * world_size,
    group=world_group,
    async_op=True,
)
work.wait()
print_rank(f"cuda all-to-all for ext_nccl_backend {q=} into {aq=}")

# this is expected to the same as nccl all-to-all-v
output_split_sizes = (
    list(chain(*([[2,1]] * (world_size//2)))) if rank < world_size - 1 else list(chain(*([[2,world_size+1]] * (world_size//2))))
)
input_split_sizes = (
    ([2] * world_size) if rank % 2 == 0 else ([1] * (world_size-1) + [world_size+1])
)
work = dist.all_to_all_single(
    output=avq,
    input=q,
    output_split_sizes=output_split_sizes,
    input_split_sizes=input_split_sizes,
    group=world_group,
    async_op=True,
)
work.wait()
print_rank(f"cuda all-to-all-v for ext_nccl_backend {q=} into {avq=}")

# this is expected to a dummy all-gather
# that sets output to all zeros and print a message
work = dummy_all_gather_into_tensor(
    output_tensor=gp,
    input_tensor=p,
    group=world_group,
    async_op=True,
)
work.wait()
print_rank(f"cuda dummy all-gather for ext_nccl_backend {p=} into {gp=}")


# this is expected to the same as nccl all-to-all-v
# except using ext_collective with some customized messages
work = extended_all_to_all_single(
    output=avq_ext,
    input=q,
    output_split_sizes=output_split_sizes,
    input_split_sizes=input_split_sizes,
    group=world_group,
    async_op=True,
)
work.wait()
print_rank(f"cuda extended all-to-all-v for ext_nccl_backend {q=} into {avq_ext=}")


# this is expected to work as a group cast
work = group_cast_collective(
    input=gc_input_tensor,
    output=gc_output_tensor,
    input_split_size_list=gc_input_split_size_list,
    output_split_size_list=gc_output_split_size_list,
    dst_indices_list=dst_indices_list,
    src_index_list=src_index_list,
    group=world_group,
    async_op=True,
)
work.wait()
assert torch.allclose(gc_output_tensor, gc_expected_tensor), (
    f"output_tensor {gc_output_tensor=} is not close to expected_tensor {gc_expected_tensor=}"
)
print_rank(f"cuda group cast for ext_nccl_backend {gc_input_tensor=} into {gc_output_tensor=}, expected {gc_expected_tensor=}")


# this is expected to work as a group reduce
work = group_reduce_collective(
    input=gr_input_tensor,
    output=gr_output_tensor,
    input_split_size_list=gr_input_split_size_list,
    output_split_size_list=gr_output_split_size_list,
    dst_index_list=dst_index_list,
    src_indices_list=src_indices_list,
    group=world_group,
    async_op=True,
)
work.wait()
assert torch.allclose(gr_output_tensor, gr_expected_tensor), (
    f"output_tensor {gr_output_tensor=} is not close to expected_tensor {gr_expected_tensor=}"
)
print_rank(f"cuda group reduce for ext_nccl_backend {gr_input_tensor=} into {gr_output_tensor=}, expected {gr_expected_tensor=}")



# --- try multi-stream and profiling --- #

dist.barrier()
torch.cuda.synchronize()

side_stream = torch.cuda.Stream()
print(f"[RANK {rank}] {side_stream=} | {side_stream.stream_id=} | {side_stream.device_index=} | {side_stream.device_type=}")

nccl_stream = backend.nccl_stream
print(f"[RANK {rank}] {nccl_stream=} | {nccl_stream.stream_id=} | {nccl_stream.device_index=} | {nccl_stream.device_type=}")

# init data shape
# use large size for profiling to avoid cpu bound
m, n, k = 16384, 16384, 8192
nh, hd = m, n 

a = torch.randn(m, k, device=device)
b = torch.randn(k, n, device=device)
s = torch.randn((m,n), device=device, dtype=torch.float32)
g = torch.empty((m*world_size, n), device=device, dtype=torch.float32)

gc_inp = gc_input_tensor_per_rank[rank].repeat_interleave(nh*hd).view(-1, nh, hd)
gc_out_exp = gc_expected_tensor_per_rank[rank].repeat_interleave(nh*hd).view(-1, nh, hd)
gc_out = torch.empty_like(gc_out_exp, dtype=dtype, device=device)

gr_inp = gr_input_tensor_per_rank[rank].repeat_interleave(nh*hd).view(-1, nh, hd)
gr_out_exp = gr_expected_tensor_per_rank[rank].repeat_interleave(nh*hd).view(-1, nh, hd)
gr_out = gr_output_tensor_per_rank[rank].repeat_interleave(nh*hd).view(-1, nh, hd)


profile_mode = os.environ.get("EXAMPLE_PROFILE_MODE", "0") == "1"
if profile_mode:
    prof_iters, prof_start_iter, prof_end_iter = 10, 5, 8
else:
    prof_iters, prof_start_iter, prof_end_iter = 1, 0, 0

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
        ag_work = dist.all_gather_into_tensor(
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

    with nvtx.add_nvtx_event("nccl stream group-cast"):
        gc_work = group_cast_collective(
            input=gc_inp,
            output=gc_out,
            input_split_size_list=gc_input_split_size_list,
            output_split_size_list=gc_output_split_size_list,
            dst_indices_list=dst_indices_list,
            src_index_list=src_index_list,
            group=world_group,
            async_op=True,
        )
    
    with nvtx.add_nvtx_event("nccl stream group-reduce"):
        gr_work = group_reduce_collective(
            input=gr_inp,
            output=gr_out,
            input_split_size_list=gr_input_split_size_list,
            output_split_size_list=gr_output_split_size_list,
            dst_index_list=dst_index_list,
            src_indices_list=src_indices_list,
            group=world_group,
            async_op=True,
        )

    with nvtx.add_nvtx_event("default_stream matmul"):
        e = a @ b
    
    dist.barrier()
    torch.cuda.synchronize()
        

dist.barrier()
torch.cuda.synchronize()
dist.destroy_process_group()
