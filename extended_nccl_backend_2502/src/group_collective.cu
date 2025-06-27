#ifndef USE_NCCL
#define USE_NCCL
#endif

#include "../include/group_collective.cuh"
#include <ATen/cuda/Atomic.cuh>
#include "launch_template.h"


ncclDataType_t to_nccl_data_type(c10::ScalarType type) {
    switch (type) {
        case at::kFloat:
            return ncclDataType_t::ncclFloat;
        case at::kHalf:
            return ncclDataType_t::ncclHalf;
        case at::kDouble:
            return ncclDataType_t::ncclDouble;
        case at::kLong:
            return ncclDataType_t::ncclInt64;
        case at::kInt:
            return ncclDataType_t::ncclInt;
        case at::kChar:
            return ncclDataType_t::ncclChar;
        // NOLINTNEXTLINE(*-narrowing-conversions, bugprone-branch-clone)
        case at::kByte:
            return ncclDataType_t::ncclUint8;
        case at::kBool:
            return ncclDataType_t::ncclUint8;
    #if defined(USE_ROCM)
        case at::kFloat8_e4m3fnuz:
            return ncclDataType_t::ncclUint8;
        case at::kFloat8_e5m2fnuz:
            return ncclDataType_t::ncclUint8;
    #else
        case at::kFloat8_e4m3fn:
            return ncclDataType_t::ncclUint8;
        case at::kFloat8_e5m2:
            return ncclDataType_t::ncclUint8;
    #endif
    #if HAS_NCCL_BF16_DATATYPE
        case at::kBFloat16:
            return ncclDataType_t::ncclBfloat16;
    #endif
        default:
            TORCH_CHECK(false, "Unconvertible NCCL type ", type);
    }
}

ncclComm_t to_nccl_comm(torch::cuda::nccl::ncclComm_t var) {
    return reinterpret_cast<ncclComm_t>(var);
}


#define GROUP_REDUCE_POST_PROCESS_DEV

#define GROUP_REDUCE_POST_PROCESS_TEST /* if set, no limit to grid size to test the highest HBM throughput */

#define GROUP_REDUCE_POST_PROCESS_NUM_SMS 32 /* following nccl comm kernel, which consumes 32 SMs at most */

#ifdef GROUP_REDUCE_POST_PROCESS_TEST
#define GROUP_REDUCE_POST_PROCESS_BLOCK_SIZE 512
#else
#define GROUP_REDUCE_POST_PROCESS_BLOCK_SIZE 1024 /* use the maximum block size */
#endif

// #define GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION 0 // 0: per element per thread
// #define GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION 1 // 1: per row per thread
#define GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION 2 // 2: per row per block
// #define GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION 3 // 3: per row per block but atomic add
// #define GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION 4


namespace torch::cuda::nccl {

    __device__ size_t binary_search_split(
        const int64_t* d_cu_split_size_list,
        size_t start,
        size_t end,
        size_t idx,
        size_t stride
    ) {
        size_t low = start, high = end;
        while (low < high) { // [low, high)
            size_t mid = low + (high - low) / 2;
            if (idx < d_cu_split_size_list[mid] * stride) {
                high = mid; // [low, mid)
            } else {
                low = mid + 1; // [mid + 1, high)
            }
        }
        return low - 1; // low == high
    }

    /** NOTE: this version uses each thread to process a single element 
     * but the highest HBM throughput (no limit to grid size) only reachs ~25%
     * TODO: optimize this kernel
    */
    #if GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION == 0
    template <typename scalar_t>
    __global__ void group_reduce_nccl_post_process_kernel( // repeat-interleaved range reduce
        scalar_t* recv_buffer,
        const scalar_t* repeated_recv_buffer,
        const int64_t* d_split_size_list,
        const int64_t* d_num_repeats_list,
        const int64_t* d_cu_split_size_list,
        const int64_t* d_repeated_cu_split_size_list,
        size_t seqlen,
        size_t num_splits,
        size_t stride0
    ) {
        size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        size_t num_threads_per_grid = blockDim.x * gridDim.x;
        size_t num_elements = seqlen * stride0;

        size_t split_idx = 0;
        for (auto idx = tid; idx < num_elements; idx += num_threads_per_grid) {
            // search for split idx that the current idx belongs
            split_idx = binary_search_split(
                d_cu_split_size_list,
                split_idx,
                num_splits,
                idx,
                stride0
            );

            // get the info about this split
            auto recv_split_start = d_cu_split_size_list[split_idx] * stride0;
            auto recv_split_size = d_split_size_list[split_idx] * stride0;
            auto repeated_recv_split_start = d_repeated_cu_split_size_list[split_idx] * stride0;
            auto num_repeats = d_num_repeats_list[split_idx];
            auto recv_split_offset_to_idx = idx - recv_split_start;

            // load the recv data with its ptr that the current idx needs to reduce to
            scalar_t* recv_data_ptr = (recv_buffer + idx);
            scalar_t recv_reduce_data = *recv_data_ptr;

            // reduce the recv data from the corr. position in repeated_recv_buffer
            const scalar_t* repeated_recv_data_ptr = (repeated_recv_buffer + repeated_recv_split_start + recv_split_offset_to_idx);

            #pragma unroll (8)
            for (size_t r = 0; r < num_repeats; ++r) {
                recv_reduce_data += __ldg(repeated_recv_data_ptr + r * recv_split_size);
            }

            // write the reduced data back to recv_buffer
            *recv_data_ptr = recv_reduce_data;
        }
    }
    #elif GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION == 1
    /** NOTE: this version uses each thread to process a single row 
     * but the highest HBM throughput (no limit to grid size) is too low, ~1%
    */
    template <typename scalar_t>
    __global__ void group_reduce_nccl_post_process_kernel( // repeat-interleaved range reduce
        scalar_t* recv_buffer,
        const scalar_t* repeated_recv_buffer,
        const int64_t* d_split_size_list,
        const int64_t* d_num_repeats_list,
        const int64_t* d_cu_split_size_list,
        const int64_t* d_repeated_cu_split_size_list,
        size_t seqlen,
        size_t num_splits,
        size_t stride0
    ) {
        size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
        size_t num_threads_per_grid = blockDim.x * gridDim.x;

        size_t split_idx = 0;
        for (auto row_idx = tid; row_idx < seqlen; row_idx += num_threads_per_grid) {
            // search for split idx that the current idx belongs
            split_idx = binary_search_split(
                d_cu_split_size_list,
                split_idx,
                num_splits,
                row_idx,
                1
            );

            // get the info about this split
            auto row_start = row_idx * stride0;
            auto recv_split_start = d_cu_split_size_list[split_idx] * stride0;
            auto repeated_recv_split_start = d_repeated_cu_split_size_list[split_idx] * stride0;
            auto recv_split_size = d_split_size_list[split_idx] * stride0;
            auto num_repeats = d_num_repeats_list[split_idx];
            auto recv_split_offset_to_idx = row_start - recv_split_start;

            // get the row start ptr of recv_buffer
            scalar_t* recv_data_ptr = (recv_buffer + row_start);

            // get the row start ptr of first partial split of repeated_recv_buffer
            const scalar_t* repeated_recv_data_ptr = (repeated_recv_buffer + repeated_recv_split_start + recv_split_offset_to_idx);

            // for-loop this row
            for (auto col_idx = 0; col_idx < stride0; ++col_idx) {
                // get the ptr of current col
                auto recv_data_ptr_this_col = (recv_data_ptr + col_idx);

                // load the original data of current col to be reduced to
                scalar_t recv_reduce_data = *recv_data_ptr_this_col;

                // get the corr. ptr of first partial data
                auto repeated_recv_data_ptr_this_col = (repeated_recv_data_ptr + col_idx);

                // load and reduce each corr. partial data
                #pragma unroll (8)
                for (size_t r = 0; r < num_repeats; ++r) {
                    recv_reduce_data += __ldg(repeated_recv_data_ptr_this_col + r * recv_split_size);
                }
             
                // write the reduced data back to recv_buffer
                *recv_data_ptr_this_col = recv_reduce_data;
            }
        }
    }
    #elif GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION == 2
    /** NOTE: this version uses each block to process a single row */
    template <typename scalar_t>
    __global__ void group_reduce_nccl_post_process_kernel( // repeat-interleaved range reduce
        scalar_t* recv_buffer,
        const scalar_t* repeated_recv_buffer,
        const int64_t* d_split_size_list,
        const int64_t* d_num_repeats_list,
        const int64_t* d_cu_split_size_list,
        const int64_t* d_repeated_cu_split_size_list,
        size_t seqlen,
        size_t num_splits,
        size_t stride0
    ) {
        extern __shared__ size_t shared_split_info[];

        size_t bid = blockIdx.x, tid_in_block = threadIdx.x;
        size_t num_blocks_per_grid = gridDim.x, num_threads_per_block = blockDim.x;

        size_t split_idx = 0;
        for (auto row_idx = bid; row_idx < seqlen; row_idx += num_blocks_per_grid) {
            if (!tid_in_block) {
                // only the thread 0 in this block searchs for split idx that the current idx belongs
                split_idx = binary_search_split(
                    d_cu_split_size_list,
                    split_idx,
                    num_splits,
                    row_idx,
                    1
                );
                // thread 0 gets the info about this split and writes it to shared memory
                shared_split_info[0] = row_idx * stride0; // row_start
                shared_split_info[1] = d_cu_split_size_list[split_idx] * stride0; // recv_split_start
                shared_split_info[2] = d_repeated_cu_split_size_list[split_idx] * stride0; // repeated_recv_split_start
                shared_split_info[3] = d_split_size_list[split_idx] * stride0; // recv_split_size
                shared_split_info[4] = d_num_repeats_list[split_idx]; // num_repeats
            } __syncthreads(); // all threads in this block wait for the same split info to be ready

            // get the info about this split
            auto row_start = shared_split_info[0];
            auto recv_split_start = shared_split_info[1];
            auto repeated_recv_split_start = shared_split_info[2];
            auto recv_split_size = shared_split_info[3];
            auto num_repeats = shared_split_info[4];
            auto recv_split_offset_to_idx = row_start - recv_split_start;

            // get the row start ptr of recv_buffer
            scalar_t* recv_data_ptr = (recv_buffer + row_start);

            // get the row start ptr of first partial split of repeated_recv_buffer
            const scalar_t* repeated_recv_data_ptr = (repeated_recv_buffer + repeated_recv_split_start + recv_split_offset_to_idx);

            // for-loop this row
            for (auto col_idx = tid_in_block; col_idx < stride0; col_idx += num_threads_per_block) {
                // get the ptr of current col
                auto recv_data_ptr_this_col = (recv_data_ptr + col_idx);

                // load the original data of current col to be reduced to
                scalar_t recv_reduce_data = *recv_data_ptr_this_col;

                // get the corr. ptr of first partial data
                auto repeated_recv_data_ptr_this_col = (repeated_recv_data_ptr + col_idx);

                // load and reduce each corr. partial data
                #pragma unroll (8)
                for (size_t r = 0; r < num_repeats; ++r) {
                    recv_reduce_data += __ldg(repeated_recv_data_ptr_this_col + r * recv_split_size);
                }
             
                // write the reduced data back to recv_buffer
                *recv_data_ptr_this_col = recv_reduce_data;
            }
        }
    }
    #elif GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION == 3
    /** NOTE: this version uses each block to process a single row, but using atomic add */
    template <typename scalar_t>
    __global__ void group_reduce_nccl_post_process_kernel( // repeat-interleaved range reduce
        scalar_t* recv_buffer,
        const scalar_t* repeated_recv_buffer,
        const int64_t* d_split_size_list,
        const int64_t* d_num_repeats_list,
        const int64_t* d_cu_split_size_list,
        const int64_t* d_repeated_cu_split_size_list,
        size_t seqlen,
        size_t num_splits,
        size_t stride0
    ) {
        extern __shared__ size_t shared_split_info[];

        size_t bid = blockIdx.x, tid_in_block = threadIdx.x;
        size_t num_blocks_per_grid = gridDim.x, num_threads_per_block = blockDim.x;

        size_t split_idx = 0;
        for (auto row_idx = bid; row_idx < seqlen; row_idx += num_blocks_per_grid) {
            if (!tid_in_block) {
                // only the thread 0 in this block searchs for split idx that the current idx belongs
                split_idx = binary_search_split(
                    d_cu_split_size_list,
                    split_idx,
                    num_splits,
                    row_idx,
                    1
                );
                // thread 0 gets the info about this split and writes it to shared memory
                shared_split_info[0] = row_idx * stride0; // row_start
                shared_split_info[1] = d_cu_split_size_list[split_idx] * stride0; // recv_split_start
                shared_split_info[2] = d_repeated_cu_split_size_list[split_idx] * stride0; // repeated_recv_split_start
                shared_split_info[3] = d_split_size_list[split_idx] * stride0; // recv_split_size
                shared_split_info[4] = d_num_repeats_list[split_idx]; // num_repeats
            } __syncthreads(); // all threads in this block wait for the same split info to be ready

            // get the info about this split
            auto row_start = shared_split_info[0];
            auto recv_split_start = shared_split_info[1];
            auto repeated_recv_split_start = shared_split_info[2];
            auto recv_split_size = shared_split_info[3];
            auto num_repeats = shared_split_info[4];
            auto recv_split_offset_to_idx = row_start - recv_split_start;

            // get the row start ptr of recv_buffer
            scalar_t* recv_data_ptr = (recv_buffer + row_start);

            // get the row start ptr of first partial split of repeated_recv_buffer
            const scalar_t* repeated_recv_data_ptr = (repeated_recv_buffer + repeated_recv_split_start + recv_split_offset_to_idx);

            // for-loop this row for num_repeats times
            for (auto rep_col_idx = tid_in_block; rep_col_idx < stride0 * num_repeats; rep_col_idx += num_threads_per_block) {
                // get the col idx and repeat idx
                size_t rep_idx = rep_col_idx / stride0;
                size_t col_idx = rep_col_idx % stride0;

                gpuAtomicAdd(
                    recv_data_ptr + col_idx,
                    *(repeated_recv_data_ptr + col_idx + rep_idx * recv_split_size)
                );
            }
        }
    }
    #elif GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION == 4
    /** NOTE: this version uses each block to process a single row */
    template <typename scalar_t>
    __global__ void group_reduce_nccl_post_process_kernel( // repeat-interleaved range reduce
        scalar_t* recv_buffer, // dkv
        const scalar_t* repeated_recv_buffer, // nccl recv 
        const int64_t* d_split_size_list, // [1024, 2048, 512]
        const int64_t* d_num_repeats_list, // [3, 2, 4]
        const int64_t* d_cu_split_size_list, // [0, 1024, 3072, 3584]
        const int64_t* d_repeated_cu_split_size_list, // [0, 1024*3, 1024*3 + 2048*2, 1024*3 + 2048*2 + 512*4]
        size_t seqlen,
        size_t num_splits, // [3]
        size_t stride0 // nh * hd
    ) {
        extern __shared__ size_t shared_split_info[];

        size_t bid = blockIdx.x, tid_in_block = threadIdx.x;
        size_t num_blocks_per_grid = gridDim.x, num_threads_per_block = blockDim.x;

        size_t split_idx = 0;
        for (auto row_idx = bid; row_idx < seqlen; row_idx += num_blocks_per_grid) {
            if (!tid_in_block) {
                // only the thread 0 in this block searchs for split idx that the current idx belongs
                split_idx = binary_search_split( // [0, 1, 2]
                    d_cu_split_size_list,
                    split_idx,
                    num_splits,
                    row_idx,
                    1
                );
                // thread 0 gets the info about this split and writes it to shared memory
                shared_split_info[0] = row_idx * stride0; // row_start
                shared_split_info[1] = d_cu_split_size_list[split_idx] * stride0; // recv_split_start
                shared_split_info[2] = d_repeated_cu_split_size_list[split_idx] * stride0; // repeated_recv_split_start
                shared_split_info[3] = d_split_size_list[split_idx] * stride0; // recv_split_size
                shared_split_info[4] = d_num_repeats_list[split_idx]; // num_repeats
            } __syncthreads(); // all threads in this block wait for the same split info to be ready

            // get the info about this split
            auto row_start = shared_split_info[0];
            auto recv_split_start = shared_split_info[1];
            auto repeated_recv_split_start = shared_split_info[2];
            auto recv_split_size = shared_split_info[3];
            auto num_repeats = shared_split_info[4];
            auto recv_split_offset_to_idx = row_start - recv_split_start;

            // get the row start ptr of recv_buffer
            scalar_t* recv_data_ptr = (recv_buffer + row_start);

            // get the row start ptr of first partial split of repeated_recv_buffer
            const scalar_t* repeated_recv_data_ptr = (repeated_recv_buffer + repeated_recv_split_start + recv_split_offset_to_idx);

            // for-loop this row
            for (auto col_idx = tid_in_block; col_idx < stride0; col_idx += num_threads_per_block) {
                // get the ptr of current col
                auto recv_data_ptr_this_col = (recv_data_ptr + col_idx);

                // load the original data of current col to be reduced to
                scalar_t recv_reduce_data = *recv_data_ptr_this_col;

                // get the corr. ptr of first partial data
                auto repeated_recv_data_ptr_this_col = (repeated_recv_data_ptr + col_idx);

                // load and reduce each corr. partial data
                #pragma unroll (8)
                for (size_t r = 0; r < num_repeats; ++r) {
                    recv_reduce_data += __ldg(repeated_recv_data_ptr_this_col + r * recv_split_size);
                }
             
                // write the reduced data back to recv_buffer
                *recv_data_ptr_this_col = recv_reduce_data;
            }
        }
    }
    #else
    #error "Unsupported GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION"
    
    #endif

    GroupReduceMetaInfo compute_group_reduce_meta_info(
        const c10::IntArrayRef recv_buffer_shape,
        const std::vector<int64_t>& output_split_size_list,
        const std::vector<std::vector<int64_t>>& src_indices_list,
        const int repeat_dim
    ) {
        size_t seqlen = recv_buffer_shape[repeat_dim];
        size_t num_splits = output_split_size_list.size();
        std::vector<int64_t> num_repeats_list; num_repeats_list.reserve(num_splits);
        std::vector<int64_t> cu_split_size_list; cu_split_size_list.reserve(num_splits + 1); cu_split_size_list.push_back(0);
        std::vector<int64_t> repeated_cu_split_size_list; repeated_cu_split_size_list.reserve(num_splits + 1); repeated_cu_split_size_list.push_back(0);
        std::vector<int64_t> repeated_recv_buffer_shape(recv_buffer_shape.begin(), recv_buffer_shape.end());

        int64_t repeat_dim_size = 0; int64_t max_split_size = 0;
        for (int64_t output_split_idx = 0; output_split_idx < num_splits; ++output_split_idx) {
            auto num_repeats = src_indices_list[output_split_idx].size();
            auto split_size = output_split_size_list[output_split_idx];
            auto repeat_split_size = split_size * num_repeats;
            max_split_size = std::max(max_split_size, split_size);

            num_repeats_list.push_back(num_repeats);
            cu_split_size_list.push_back(cu_split_size_list[output_split_idx] + split_size);
            repeated_cu_split_size_list.push_back(repeated_cu_split_size_list[output_split_idx] + repeat_split_size);
            repeat_dim_size += repeat_split_size;
        }
        repeated_recv_buffer_shape[repeat_dim] = repeat_dim_size;

        /** NOTE: do not wrap it to c10::ArrayRef:
         *  return c10::makeArrayRef(repeated_recv_buffer_shape);
         * since c10::ArrayRef only holds the reference which is local to this function
         * thus might resulting in dangling reference
         */
        return GroupReduceMetaInfo(
            seqlen,
            num_splits,
            repeat_dim_size,
            max_split_size,
            std::move(num_repeats_list),
            std::move(cu_split_size_list),
            std::move(repeated_cu_split_size_list),
            std::move(repeated_recv_buffer_shape)
        );
    }

    // group cast collective
    void group_cast_nccl_kernel(
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
        at::cuda::CUDAStream& stream
    ) {
        auto num_input_splits = input_split_size_list.size();
        auto num_output_splits = output_split_size_list.size();

        auto nccl_data_type = to_nccl_data_type(type);
        auto nccl_comm = to_nccl_comm(comm);

        int64_t input_offset = 0, output_offset = 0;
        NCCLCHECK(ncclGroupStart());
        for (size_t input_split_idx = 0; input_split_idx < num_input_splits; ++input_split_idx) {
            auto input_size = input_split_size_list[input_split_idx] * stride0;
            for (auto dst_rank : dst_indices_list[input_split_idx]) {
                NCCLCHECK(ncclSend(
                    (const void*) (send_buffer + input_offset * element_size),
                    input_size,
                    nccl_data_type,
                    dst_rank,
                    nccl_comm,
                    stream
                ));
            }
            input_offset += input_size;
        }
        for (size_t output_split_idx = 0; output_split_idx < num_output_splits; ++output_split_idx) {
            auto src_rank = src_index_list[output_split_idx];
            auto output_size = output_split_size_list[output_split_idx] * stride0;
            NCCLCHECK(ncclRecv(
                (void *) (recv_buffer + output_offset * element_size),
                output_size,
                nccl_data_type,
                src_rank,
                nccl_comm,
                stream
            ));
            output_offset += output_size;
        }
        NCCLCHECK(ncclGroupEnd());
    }

    // group reduce collective
    void group_reduce_nccl_kernel(
        void* send_buffer,
        void* recv_buffer,
        void* repeated_recv_buffer,
        const std::vector<int64_t>& input_split_size_list,
        const std::vector<int64_t>& output_split_size_list,
        const std::vector<int64_t>& dst_index_list,
        const std::vector<std::vector<int64_t>>& src_indices_list,
        size_t stride0,
        size_t element_size,
        c10::ScalarType type,
        ncclComm_t comm,
        at::cuda::CUDAStream& stream,
        /* for post-process kernel */
        const int64_t* d_split_size_list,
        const int64_t* d_num_repeats_list,
        const int64_t* d_cu_split_size_list,
        const int64_t* d_repeated_cu_split_size_list,
        size_t seqlen,
        size_t seqlen_r,
        size_t num_splits,
        size_t max_split_size
    ) {
        auto num_input_splits = input_split_size_list.size();
        auto num_output_splits = output_split_size_list.size();

        auto nccl_data_type = to_nccl_data_type(type);
        auto nccl_comm = to_nccl_comm(comm);

        // group-reduce kernel implemented by nccl group p2p
        int64_t input_offset = 0, output_offset = 0;
        NCCLCHECK(ncclGroupStart());
        for (size_t input_split_idx = 0; input_split_idx < num_input_splits; ++input_split_idx) {
            auto dst_rank = dst_index_list[input_split_idx];
            auto input_size = input_split_size_list[input_split_idx] * stride0;
            NCCLCHECK(ncclSend(
                (const void*) (send_buffer + input_offset * element_size),
                input_size,
                nccl_data_type,
                dst_rank,
                nccl_comm,
                stream
            ));
            input_offset += input_size;
        }
        for (size_t output_split_idx = 0; output_split_idx < num_output_splits; ++output_split_idx) {
            auto output_size = output_split_size_list[output_split_idx] * stride0;
            for (auto src_rank : src_indices_list[output_split_idx]) {
                NCCLCHECK(ncclRecv(
                    (void *) (repeated_recv_buffer + output_offset * element_size),
                    output_size,
                    nccl_data_type,
                    src_rank,
                    nccl_comm,
                    stream
                ));
                /** NOTE: since nccl recv can not handle atomic add,
                 * we have to interleavedly repeat the recv buffer
                 * for each src rank for the same split
                 * and apply a post-process reduce to sum them up into the original recv buffer
                 * and for convenience and safety, we allocate this repeated_recv_buffer outside
                 * thus we can utilize the torch's caching allocator to avoid reallocation cuda memory
                 * as well as ensure meory false-reuse using workNccl's stashed_for_allocator_safety_
                 */
                output_offset += output_size;
            }
        }
        NCCLCHECK(ncclGroupEnd());

        // post-process reduce kernel from repeated_recv_buffer to recv_buffer
        /** NOTE: we don't want the post-process kernel occupies too many SMs
         * thus we set a small fixed grid size, and only relax the limit when testing
         */
        #ifdef GROUP_REDUCE_POST_PROCESS_TEST
        dim3 gridDims((seqlen * stride0 + GROUP_REDUCE_POST_PROCESS_BLOCK_SIZE - 1) / GROUP_REDUCE_POST_PROCESS_BLOCK_SIZE);
        #else
        dim3 gridDims(GROUP_REDUCE_POST_PROCESS_NUM_SMS);
        #endif
        dim3 blockDims(GROUP_REDUCE_POST_PROCESS_BLOCK_SIZE);

        #if GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION >= 2
        size_t sharedMemSize = 5 * sizeof(size_t);
        #else
        size_t sharedMemSize = 0;
        #endif

        #ifdef GROUP_REDUCE_POST_PROCESS_DEV
        group_reduce_nccl_post_process_cute_kernel<cutlass::bfloat16_t, 128>(
            static_cast<cutlass::bfloat16_t*>(recv_buffer),
            static_cast<cutlass::bfloat16_t*>(repeated_recv_buffer),
            seqlen,
            stride0,
            seqlen_r,
            num_splits,
            max_split_size,
            d_cu_split_size_list, // cu_split_size_o,
            d_split_size_list, // split_size_list,
            d_repeated_cu_split_size_list, // cu_split_size_r,
            d_num_repeats_list, // num_repeats_list,
            stream.stream()
        );
        #else
        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, /* add float16/bfloat16 to dispatch types */
            type,
            #if GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION == 0
            "group_reduce_nccl_post_process_kernel_v0",
            #elif GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION == 1
            "group_reduce_nccl_post_process_kernel_v1",
            #elif GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION == 2
            "group_reduce_nccl_post_process_kernel_v2",
            #elif GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION == 3
            "group_reduce_nccl_post_process_kernel_v3",
            #elif GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION == 4
            "group_reduce_nccl_post_process_kernel_v4",
            #else
            #error "Unsupported GROUP_REDUCE_POST_PROCESS_KERNEL_VERSION"
            #endif
            [&] {
            group_reduce_nccl_post_process_kernel<scalar_t> /* auto-deduced `scalar_t` by the macro */
                /** NOTE: the post-process kernel is supposed to run 
                 * on the same stream as the group reduce kernel, i.e. nccl stream
                 */
                <<<gridDims, blockDims, sharedMemSize, stream.stream()>>>(
                    static_cast<scalar_t*>(recv_buffer),
                    static_cast<const scalar_t*>(repeated_recv_buffer),
                    d_split_size_list,
                    d_num_repeats_list,
                    d_cu_split_size_list,
                    d_repeated_cu_split_size_list,
                    seqlen,
                    num_splits,
                    stride0
                );
            }
        );
        #endif

    }

} // namespace torch::cuda::nccl
