from datetime import timedelta

import torch
import torch.distributed as dist


class GroupCastOptions:
    timeout: timedelta
    asyncOp: bool
    

class GroupReduceOptions:
    timeout: timedelta
    asyncOp: bool


class MagiNCCLBackend(dist.ProcessGroupNCCL):
    
    def __init__(
        self,
        store: dist.Store,
        rank: int,
        size: int,
    ) -> None: ...
    
    @property
    def nccl_stream(self) -> torch.cuda.Stream: ...
    
    def _dummy_allgather_base(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        opts=...,
    ) -> dist.Work: ...
    
    def extended_alltoall_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts=...,
    ) -> dist.Work: ...
    
    def group_cast(
        input: torch.Tensor,
        output: torch.Tensor,
        input_split_size_list: list[int],
        output_split_size_list: list[int],
        dst_indices_list: list[list[int]],
        src_index_list: list[int],
        opts=...,
        **kwargs,
    ) -> dist.Work: ...
    
    def group_reduce(
        input: torch.Tensor,
        output: torch.Tensor,
        input_split_size_list: list[int],
        output_split_size_list: list[int],
        dst_index_list: list[int],
        src_indices_list: list[list[int]],
        opts=...,
        **kwargs,
    ) -> dist.Work: ...