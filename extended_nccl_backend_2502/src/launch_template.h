#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"  // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/kernel_launch.h>

#include "static_switch.h"
#include "reduce_add_kernel.h"

#include <assert.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                        \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())


using namespace cute;

template<typename T_out, uint32_t kBlockM, uint32_t kBlockN>
void run_fast_range_reduce(
    T_out* ptr_O,
    T_out* ptr_R,

    int64_t seqlen,
    int64_t hidden_size,
    int64_t seqlen_r,
    int64_t num_splits,
    int64_t max_split_size,

    const int64_t* cu_split_size_o,
    const int64_t* split_size_list,
    const int64_t* cu_split_size_r,
    const int64_t* num_repeats_list,

    cudaStream_t stream
) {
    using ArchTag = cutlass::arch::Sm90;
    using RangeReduceKernel = FastRangeReduceKernel<T_out, kBlockM, kBlockN, ArchTag>;

    auto kernel_params = RangeReduceKernel::to_underlying_arguments({
        ptr_O,
        {seqlen, hidden_size},
        {hidden_size, _1{}},
        ptr_R,
        {seqlen_r, hidden_size},
        {hidden_size, _1{}},
        num_splits,
        cu_split_size_o,
        split_size_list,
        cu_split_size_r,
        num_repeats_list,
        max_split_size
    });

    dim3 grid_dims = RangeReduceKernel::get_grid_shape(kernel_params);
    dim3 block_dims = RangeReduceKernel::get_block_shape();

    auto kernel = cutlass::device_kernel<RangeReduceKernel>;
    int smem_size = RangeReduceKernel::SharedStorageSize;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    cutlass::kernel_launch<RangeReduceKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/);
    CHECK_CUDA_KERNEL_LAUNCH();
}


template<typename T_out, uint32_t kBlockN>
void group_reduce_nccl_post_process_cute_kernel(
    T_out* ptr_O,
    T_out* ptr_R,

    int64_t seqlen,
    int64_t hidden_size,
    int64_t seqlen_r,
    int64_t num_splits,
    int64_t max_split_size,

    const int64_t* cu_split_size_o,
    const int64_t* split_size_list,
    const int64_t* cu_split_size_r,
    const int64_t* num_repeats_list,

    cudaStream_t stream
) {
    // TODO: tuning block size
    static constexpr uint32_t kBlockM = 128;
    run_fast_range_reduce<T_out, kBlockM, kBlockN>(
        ptr_O,
        ptr_R,
        seqlen,
        hidden_size,
        seqlen_r,
        num_splits,
        max_split_size,
        cu_split_size_o,
        split_size_list,
        cu_split_size_r,
        num_repeats_list,
        stream
    );
}
