import torch
import torch.distributed as dist


class ExtProcessGroupNCCL(dist.ProcessGroupNCCL):
    
    def __init__(
        self,
        store: dist.Store,
        rank: int,
        size: int,
    ) -> None: ...
    
    @property
    def nccl_stream(self) -> torch.cuda.Stream: ...