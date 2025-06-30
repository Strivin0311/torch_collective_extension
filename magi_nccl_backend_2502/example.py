import os
from typing import cast
from itertools import chain

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroupNCCL

import magi_nccl
from magi_nccl import MagiNCCLBackend
from magi_nccl_interface import (
    group_cast_collective,
    group_reduce_collective,
)
from utils import (
    print_rank,
    sanity_check_for_group_cast_meta_args_per_rank,
    sanity_check_for_group_reduce_meta_args_per_rank,
    get_group_reduce_post_process_bytes,
)
import nvtx


# get some env variable as flags
profile_mode = os.environ.get("EXAMPLE_PROFILE_MODE", "0") == "1"
use_ncu_for_profile = os.environ.get("EXAMPLE_USE_NCU_FOR_PROFILE", "0") == "1"

# init process group
dist.init_process_group(
    backend="cpu:gloo,cuda:magi_nccl",
    # backend="magi_nccl", # NOTE: magi_nccl is not supported for cpu
)

# get rank, world_size and init device
rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
device = torch.cuda.current_device()
dtype = torch.bfloat16

# just print the function name to see if it is loaded
print_rank(f"{magi_nccl.createMagiNCCLBackend=}")


# --- init pg and backend --- #

# get the process group backend
world_group = dist.group.WORLD
print_rank(f"WorldGroup: {type(world_group)=}, {world_group._get_backend_name()=}")

backend: dist.Backend = world_group._get_backend(torch.device(device))
print_rank(f"WorldGroup: {type(backend)=}")
assert isinstance(backend, MagiNCCLBackend), (
    f"expected MagiNCCLBackend, got {type(backend)=}"
)
backend: MagiNCCLBackend = cast(MagiNCCLBackend, backend)
assert not isinstance(backend, ProcessGroupNCCL), (
    f"We expect MagiNCCLBackend not as a subclass of ProcessGroupNCCL"
)

pg = dist.new_group(list(range(world_size)), backend="magi_nccl")
print_rank(f"NewGroup: {type(pg)=}, {pg._get_backend_name()=}")

pg_backend: dist.Backend = pg._get_backend(torch.device(device))
print_rank(f"NewGroup: {type(pg_backend)=}")
assert isinstance(pg_backend, MagiNCCLBackend), (
    f"expected MagiNCCLBackend, got {type(pg_backend)=}"
)


# --- try simple functionalities --- #

ar_ans = world_size * (world_size - 1) // 2

x = torch.zeros(world_size, dtype=dtype) + rank
arx = x.clone()
arx_exp = torch.full_like(x, ar_ans)

y = x.to(device)
ary = y.clone()
ary_exp = torch.full_like(y, ar_ans)

z = y.clone()
bz = z.clone()
bz_exp = torch.zeros_like(z)

p = torch.arange(world_size, device=device, dtype=dtype) + rank * 2
gp = torch.empty(world_size**2, device=device, dtype=dtype)
gp_exp = torch.concat([
    torch.arange(world_size, device=device, dtype=dtype) + r * 2
    for r in range(world_size)
], dim=0)

q = torch.arange(world_size*2, device=device, dtype=dtype) + rank * 2
aq = torch.empty(world_size*2, device=device, dtype=dtype)
aq_exp = q.clone()

avq = torch.empty(
    (3 * (world_size // 2)) 
    if rank < world_size - 1 
    else ((world_size + 3) * (world_size // 2)),
    device=device, 
    dtype=dtype
)
avq_exp = [
    torch.tensor([0., 1., 2., 4., 5., 6.], device=device, dtype=dtype),
    torch.tensor([2., 3., 3., 6., 7., 7.], device=device, dtype=dtype),
    torch.tensor([4., 5., 4., 8., 9., 8.], device=device, dtype=dtype),
    torch.tensor([6.,  7.,  5.,  6.,  7.,  8.,  9., 10., 11.,  9., 10., 11., 12., 13.], device=device, dtype=dtype),
    
][rank]

output_split_sizes = (
    list(chain(*([[2,1]] * (world_size//2)))) 
    if rank < world_size - 1 
    else list(chain(*([[2,world_size+1]] * (world_size//2))))
)
input_split_sizes = (
    ([2] * world_size) 
    if rank % 2 == 0 
    else ([1] * (world_size-1) + [world_size+1])
)

# NOTE: we cannot fetch the nccl stream at this point
# since both the nccl stream and nccl comm are 
# lazily initialized until the first collective call
try:
    backend.nccl_stream
except RuntimeError as e:
    print_rank(f"{e=}")

# this goes through gloo backend
dist.all_reduce(arx, group=world_group)
print_rank(f"cpu all-reduce for gloo backend from {x=} to {arx=}")
assert torch.allclose(arx, arx_exp)

# this goes through nccl backend
# and is expected to the same as nccl all-reduce
dist.all_reduce(ary, group=world_group)  # the result should be [ans] * size
print_rank(f"cuda all-reduce for magi_nccl from {y=} to {ary=}")
assert torch.allclose(ary, ary_exp)

# this is expected to the same as nccl broadcast
dist.broadcast(bz, 0, group=pg) # the result should be [0] * size
print_rank(f"cuda broadcast for magi_nccl from {z=} to {bz=}")
assert torch.allclose(bz, bz_exp)

# this is expected to the same as nccl all-gather
work = dist.all_gather_into_tensor(
    output_tensor=gp,
    input_tensor=p,
    group=world_group,
    async_op=True,
)
work.wait()
print_rank(f"cuda all-gather for magi_nccl {p=} into {gp=}")
assert torch.allclose(gp, gp_exp)

# this is expected to the same as nccl all-to-all for list of tensors
input = torch.arange(4, device=device, dtype=dtype) + rank * 4
input = list(input.chunk(4))
output = list(torch.empty([4], device=device, dtype=dtype).chunk(4))
exp_output = list(torch.arange(4, device=device, dtype=dtype) * world_size + rank)
work = dist.all_to_all(output, input, group=world_group, async_op=True)
work.wait()
print_rank(f"cuda all-to-all-list for magi_nccl {input=} into {output=}")
assert all(torch.allclose(o, e) for o, e in zip(output, exp_output))


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
print_rank(f"cuda all-to-all-single for magi_nccl {q=} into {aq=}")
assert torch.allclose(aq, aq_exp)

# this is expected to the same as nccl all-to-all-v
work = dist.all_to_all_single(
    output=avq,
    input=q,
    output_split_sizes=output_split_sizes,
    input_split_sizes=input_split_sizes,
    group=world_group,
    async_op=True,
)
work.wait()
print_rank(f"cuda all-to-all-single-v for magi_nccl {q=} into {avq=}")


# --- try nccl stream --- #

nccl_stream = backend.nccl_stream
print_rank(f"{rank}] {nccl_stream=} | {nccl_stream.stream_id=} | {nccl_stream.device_index=} | {nccl_stream.device_type=}")


# --- try group cast --- #

nh, hd = 16, 128 # simulate 3 * 4k = 12k seqlen of kv
sunit = 1024 # seqlen unit for "1" in the split_size_list, to simulate long seqlen

# init tensor
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
gc_inp = gc_input_tensor_per_rank[rank].repeat_interleave(sunit*nh*hd).view(-1, nh, hd)
gc_out_exp = gc_expected_tensor_per_rank[rank].repeat_interleave(sunit*nh*hd).view(-1, nh, hd)
gc_out = torch.empty_like(gc_out_exp, dtype=dtype, device=device)

# init meta
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
    [2, 2, 1, 1, 1], # r1
    [1, 1, 2, 2], # r2
    [1, 1, 1, 1], # r3
]
src_index_list_per_rank = [
    [1, 2, 3], # r0
    [0, 2, 0, 3, 3], # r1
    [0, 0, 1, 3], # r2
    [1, 1, 2, 2] # r3
]
gc_input_split_size_list = gc_input_split_size_list_per_rank[rank]
gc_output_split_size_list = gc_output_split_size_list_per_rank[rank]
gc_input_split_size_list = list(map(lambda x: x * sunit, gc_input_split_size_list))
gc_output_split_size_list = list(map(lambda x: x * sunit, gc_output_split_size_list))
dst_indices_list = dst_indices_list_per_rank[rank]
src_index_list = src_index_list_per_rank[rank]

# sanity check
sanity_check_for_group_cast_meta_args_per_rank(
    input_split_size_list_per_rank=gc_input_split_size_list_per_rank,
    output_split_size_list_per_rank=gc_output_split_size_list_per_rank,
    dst_indices_list_per_rank=dst_indices_list_per_rank,
    src_index_list_per_rank=src_index_list_per_rank,
    world_size=world_size,
    check_nccl_send_recv=True,
)

# run group cast
work = group_cast_collective(
    input=gc_inp,
    output=gc_out,
    input_split_size_list=gc_input_split_size_list,
    output_split_size_list=gc_output_split_size_list,
    dst_indices_list=dst_indices_list,
    src_index_list=src_index_list,
    group=world_group,
    async_op=True,
)

# check result
work.wait()
print_rank(f"For group cast, {gc_out=} is expected to be all close to {gc_out_exp=}")
assert torch.allclose(gc_out, gc_out_exp)


# --- try group reduce --- #

# init tensor
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
gr_inp = gr_input_tensor_per_rank[rank].repeat_interleave(sunit*nh*hd).view(-1, nh, hd)
gr_out_exp = gr_expected_tensor_per_rank[rank].repeat_interleave(sunit*nh*hd).view(-1, nh, hd)
gr_out = gr_output_tensor_per_rank[rank].repeat_interleave(sunit*nh*hd).view(-1, nh, hd)

# init meta
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
gr_input_split_size_list = gr_input_split_size_list_per_rank[rank]
gr_output_split_size_list = gr_output_split_size_list_per_rank[rank]
gr_input_split_size_list = list(map(lambda x: x * sunit, gr_input_split_size_list))
gr_output_split_size_list = list(map(lambda x: x * sunit, gr_output_split_size_list))
dst_index_list = dst_index_list_per_rank[rank]
src_indices_list = src_indices_list_per_rank[rank]

# sanity check
sanity_check_for_group_reduce_meta_args_per_rank(
    input_split_size_list_per_rank=gr_input_split_size_list_per_rank,
    output_split_size_list_per_rank=gr_output_split_size_list_per_rank,
    dst_index_list_per_rank=dst_index_list_per_rank,
    src_indices_list_per_rank=src_indices_list_per_rank,
    world_size=world_size,
    check_nccl_send_recv=True,
)

gr_post_process_bytes = get_group_reduce_post_process_bytes(
    output_shape=gr_out.shape,
    output_split_size_list=gr_output_split_size_list,
    src_indices_list=src_indices_list,
    dtype=dtype,
)

work = group_reduce_collective(
    input=gr_inp,
    output=gr_out,
    input_split_size_list=gr_input_split_size_list,
    output_split_size_list=gr_output_split_size_list,
    dst_index_list=dst_index_list,
    src_indices_list=src_indices_list,
    group=world_group,
    async_op=True,
)
work.wait()
print_rank(f"For group-reduce (with {gr_post_process_bytes=}), {gr_out=} is expected to be all close to {gr_out_exp=}")
assert torch.allclose(gr_out, gr_out_exp)


# --- try multi-stream and profiling --- #

dist.barrier()
torch.cuda.synchronize()

side_stream = torch.cuda.Stream()
print_rank(f"{side_stream=} | {side_stream.stream_id=} | {side_stream.device_index=} | {side_stream.device_type=}")

# init data shape
# use large size for profiling to avoid cpu bound
m, n, k = 16384, 16384, 8192

a = torch.randn(m, k, device=device)
b = torch.randn(k, n, device=device)
s = torch.randn((m,n), device=device, dtype=dtype)
g = torch.empty((m*world_size, n), device=device, dtype=dtype)

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
    
    with nvtx.add_nvtx_event(f"rank{rank} nccl_stream allgather"):
        ag_work = dist.all_gather_into_tensor(
            g,
            s,
            group=world_group,
            async_op=True
        )
        
    side_stream.wait_stream(torch.cuda.default_stream())
    with nvtx.add_nvtx_event(f"rank{rank} side_stream matmul"):
        with torch.cuda.stream(side_stream):
            c = a @ b
    
    nccl_stream.wait_stream(torch.cuda.default_stream())
    with nvtx.add_nvtx_event(f"rank{rank} nccl_stream matmul"):
        with torch.cuda.stream(nccl_stream):
                d = a @ b

    with nvtx.add_nvtx_event(f"rank{rank} nccl stream group-cast"):
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
    
    with nvtx.add_nvtx_event(f"rank{rank} nccl stream group-reduce"):
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

    with nvtx.add_nvtx_event(f"rank{rank} default_stream matmul"):
        e = a @ b
    
    dist.barrier()
    torch.cuda.synchronize()
        

dist.barrier()
torch.cuda.synchronize()
dist.destroy_process_group()
