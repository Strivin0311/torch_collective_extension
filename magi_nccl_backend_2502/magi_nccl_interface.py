import torch
import torch.distributed as dist
from torch.distributed.c10d_logger import _exception_logger
from torch.distributed.distributed_c10d import (
    _check_single_tensor, 
    _rank_not_in_group, 
    _warn_not_in_group, 
    _get_default_group,
    _ensure_all_tensors_same_dtype,
)

from magi_nccl import MagiNCCLBackend


@_exception_logger
def group_cast_collective(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_indices_list: list[list[int]],
    src_index_list: list[int],
    group: dist.Backend = None,
    async_op: bool = False,
): # -> dist.Work | None
    """TODO: add docstring"""
    if _rank_not_in_group(group):
        _warn_not_in_group("group_cast")
        return

    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")
    _ensure_all_tensors_same_dtype(output, input)

    if input.is_complex():
        input = torch.view_as_real(input)
    if output.is_complex():
        output = torch.view_as_real(output)

    group = group or _get_default_group()
    
    backend = group._get_backend(torch.device("cuda"))
    assert isinstance(backend, MagiNCCLBackend), (
        f"expected MagiNCCLBackend, got {type(group)=}"
    )
    
    work = backend.group_cast(
        input,
        output,
        input_split_size_list,
        output_split_size_list,
        dst_indices_list,
        src_index_list,
    )

    if async_op:
        return work
    else:
        work.wait()
        
        
@_exception_logger
def group_reduce_collective(
    input: torch.Tensor,
    output: torch.Tensor,
    input_split_size_list: list[int],
    output_split_size_list: list[int],
    dst_index_list: list[int],
    src_indices_list: list[list[int]],
    group: dist.Backend = None,
    async_op: bool = False,
): # -> dist.Work | None
    """TODO: add docstring"""
    if _rank_not_in_group(group):
        _warn_not_in_group("group_cast")
        return

    _check_single_tensor(output, "output")
    _check_single_tensor(input, "input")
    _ensure_all_tensors_same_dtype(output, input)

    if input.is_complex():
        input = torch.view_as_real(input)
    if output.is_complex():
        output = torch.view_as_real(output)

    group = group or _get_default_group()
    
    backend = group._get_backend(torch.device("cuda"))
    assert isinstance(backend, MagiNCCLBackend), (
        f"expected MagiNCCLBackend, got {type(group)=}"
    )
    
    work = backend.group_reduce(
        input,
        output,
        input_split_size_list,
        output_split_size_list,
        dst_index_list,
        src_indices_list,
    )

    if async_op:
        return work
    else:
        work.wait()