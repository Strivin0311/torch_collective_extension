import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    AllgatherOptions
)


class ExtProcessGroupNCCL(dist.ProcessGroupNCCL):
    
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