#pragma once

#ifndef USE_NCCL
#define USE_NCCL
#endif

#include <nccl.h>
#include <torch/csrc/cuda/nccl.h>


namespace torch::cuda::nccl {

    #define NCCLCHECK(cmd) do {                             \
        ncclResult_t res = cmd;                             \
        if (res != ncclSuccess) {                           \
            printf(                                         \
                "Failed, NCCL Error: %s:%d '%s'\n",         \
                __FILE__, __LINE__, ncclGetErrorString(res) \
            );                                              \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)


    TORCH_CUDA_CPP_API void group_cast_nccl_kernel(
        void* send_buffer,
        void* recv_buffer,
        const std::vector<int64_t>& input_split_size_list,
        const std::vector<int64_t>& output_split_size_list,
        const std::vector<std::vector<int64_t>>& dst_indices_list,
        const std::vector<int64_t>& src_index_list,
        size_t stride0,
        size_t element_size,
        c10::ScalarType type,
        ncclComm_t comm,
        at::cuda::CUDAStream& stream);
} // namespace torch::cuda::nccl