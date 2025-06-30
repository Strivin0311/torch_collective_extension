from datetime import timedelta

import torch
import torch.distributed as dist


class GroupCastOptions:
    timeout: timedelta
    asyncOp: bool
    

class GroupReduceOptions:
    timeout: timedelta
    asyncOp: bool


# NOTE: here we set MagiNCCLBackend as a subclass of ProcessGroupNCCL
# to inherit the same common interfaces
# however, it is NOT true in both python end and c++ end
# i.e. isinstance(MagiNCCLBackend, ProcessGroupNCCL) is False
class MagiNCCLBackend(dist.ProcessGroupNCCL):
    
    def __init__(
        self,
        store: dist.Store,
        rank: int,
        size: int,
    ) -> None: ...
    
    @property
    def nccl_stream(self) -> torch.cuda.Stream: ...
    
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