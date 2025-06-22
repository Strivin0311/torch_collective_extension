#pragma once

#include <torch/csrc/cuda/nccl.h>


namespace torch::cuda::nccl {
    TORCH_CUDA_CPP_API void all2all_single_unequal_split(
        void* sendbuff,
        const size_t* sendcounts,
        const size_t* senddispls,
        void* recvbuff,
        const size_t* recvcounts,
        const size_t* recvdispls,
        size_t size,
        c10::ScalarType type,
        ncclComm_t comm,
        at::cuda::CUDAStream& stream);
} // namespace torch::cuda::nccl