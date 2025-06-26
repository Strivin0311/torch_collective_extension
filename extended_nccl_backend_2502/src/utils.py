import math
from itertools import chain

import torch

def _sanity_check_nccl_send_recv(
    num_send_list: list[int],
    num_recv_list: list[int],
    world_size: int
):
    if num_send_list != num_recv_list:
        num_diff_idxs: list[int] = torch.nonzero(
            torch.tensor(num_send_list) - torch.tensor(num_recv_list), 
            as_tuple=True
        )[0].tolist()
        
        msg = [
            (
                "For each pair of src_rank and dst_rank, "
                "The number of nccl send calls launched by src_rank for dst_rank "
                "should be identical to the number of nccl recv calls launched by dst_rank for src_rank, "
                "but got: "
            )
        ]
        for idx in num_diff_idxs:
            src_rank, dst_rank = divmod(idx, world_size)
            msg.append(
                f"The number of send calls launched by rank{src_rank} for rank{dst_rank} is {num_send_list[idx]} "
                f"while the number of recv calls launched by rank{dst_rank} for rank{src_rank} is {num_recv_list[idx]}."
            )
            
        raise AssertionError("\n".join(msg))


def sanity_check_for_group_cast_meta_args_per_rank(
    input_split_size_list_per_rank: list[list[int]],
    output_split_size_list_per_rank: list[list[int]],
    dst_indices_list_per_rank: list[list[list[int]]],
    src_index_list_per_rank: list[list[int]],
    world_size: int,
    check_nccl_send_recv: bool = False,
) -> None:
    for rank in range(world_size):
        # sanity check for shape
        input_split_size_list = input_split_size_list_per_rank[rank]
        output_split_size_list = output_split_size_list_per_rank[rank]
        dst_indices_list = dst_indices_list_per_rank[rank]
        src_index_list = src_index_list_per_rank[rank]
        assert len(input_split_size_list) == len(dst_indices_list), (
            f"input_split_size_list and dst_indices_list should have the same length, "
            f"but got {len(input_split_size_list)=} and {len(dst_indices_list)=}"
        )
        assert len(output_split_size_list) == len(src_index_list), (
            f"output_split_size_list and src_index_list should have the same length, "
            f"but got {len(output_split_size_list)=} and {len(src_index_list)=}"
        )
    
        # sanity check for rank value
        assert all(0 <= dst_rank < world_size for dst_rank in chain(*dst_indices_list)), (
            f"dst_indices_list should contain ranks in [0, {world_size - 1}], "
            f"but got {dst_indices_list=}"
        )
        assert all(0 <= src_rank < world_size for src_rank in src_index_list), (
            f"src_index_list should contain ranks in [0, {world_size - 1}], "
            f"but got {src_index_list=}"
        )

    # sanity check for nccl send/recv consistent number of calls
    if check_nccl_send_recv:
        # num_send[src_rank*world_size + dst_rank]: the number of nccl send calls launched by src_rank for dst_rank
        num_send_list: list[int] = [0] * world_size**2
        # num_recv[src_rank*world_size + dst_rank]: the number of nccl recv calls launched by dst_rank for src_rank
        num_recv_list: list[int] = [0] * world_size**2
        
        for rank in range(world_size):
            src_rank = rank
            dst_indices_list = dst_indices_list_per_rank[src_rank]
            for dst_rank in chain(*dst_indices_list):
                num_send_list[src_rank*world_size + dst_rank] += 1
            
            dst_rank = rank
            src_index_list = src_index_list_per_rank[dst_rank]
            for src_rank in src_index_list:
                num_recv_list[src_rank*world_size + dst_rank] += 1
                
        _sanity_check_nccl_send_recv(
            num_send_list=num_send_list,
            num_recv_list=num_recv_list,
            world_size=world_size,
        )


def sanity_check_for_group_reduce_meta_args_per_rank(
    input_split_size_list_per_rank: list[list[int]],
    output_split_size_list_per_rank: list[list[int]],
    dst_index_list_per_rank: list[list[int]],
    src_indices_list_per_rank: list[list[list[int]]],
    world_size: int,
    check_nccl_send_recv: bool = False,
) -> None:
    for rank in range(world_size):
        # sanity check for shape
        input_split_size_list = input_split_size_list_per_rank[rank]
        output_split_size_list = output_split_size_list_per_rank[rank]
        dst_index_list = dst_index_list_per_rank[rank]
        src_indices_list = src_indices_list_per_rank[rank]
        assert len(input_split_size_list) == len(dst_index_list), (
            f"input_split_size_list and dst_index_list should have the same length, "
            f"but got {len(input_split_size_list)=} and {len(dst_index_list)=}"
        )
        assert len(output_split_size_list) == len(src_indices_list), (
            f"output_split_size_list and src_indices_list should have the same length, "
            f"but got {len(output_split_size_list)=} and {len(src_indices_list)=}"
        )
    
        # sanity check for rank value
        assert all(0 <= dst_rank < world_size for dst_rank in dst_index_list), (
            f"dst_index_list should contain ranks in [0, {world_size - 1}], "
            f"but got {dst_index_list=}"
        )
        assert all(0 <= src_rank < world_size for src_rank in chain(*src_indices_list)), (
            f"src_indices_list should contain ranks in [0, {world_size - 1}], "
            f"but got {src_indices_list=}"
        )

    # sanity check for nccl send/recv consistent number of calls
    if check_nccl_send_recv:
        # num_send[src_rank*world_size + dst_rank]: the number of nccl send calls launched by src_rank for dst_rank
        num_send_list: list[int] = [0] * world_size**2
        # num_recv[src_rank*world_size + dst_rank]: the number of nccl recv calls launched by dst_rank for src_rank
        num_recv_list: list[int] = [0] * world_size**2
        
        for rank in range(world_size):
            src_rank = rank
            dst_index_list = dst_index_list_per_rank[src_rank]
            for dst_rank in dst_index_list:
                num_send_list[src_rank*world_size + dst_rank] += 1
            
            dst_rank = rank
            src_indices_list = src_indices_list_per_rank[dst_rank]
            for src_rank in chain(*src_indices_list):
                num_recv_list[src_rank*world_size + dst_rank] += 1
        
        _sanity_check_nccl_send_recv(
            num_send_list=num_send_list,
            num_recv_list=num_recv_list,
            world_size=world_size,
        )


def get_group_reduce_post_process_bytes(
    output_shape: list[int],
    output_split_size_list: list[int],
    src_indices_list: list[list[int]],
    dtype: torch.dtype,
) -> int:
    seqlen = output_shape[0]
    stride0 = math.prod(output_shape[1:])
    repeated_seqlen = sum([
        split_size * len(src_indices)
        for split_size, src_indices in zip(output_split_size_list, src_indices_list)]
    )
    
    num_loads = (seqlen + repeated_seqlen) * stride0
    num_save = seqlen * stride0
    
    return (num_loads + num_save) * dtype.itemsize
    
