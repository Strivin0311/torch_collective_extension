#ifndef USE_NCCL
#define USE_NCCL
#endif

#include "../include/group_collective.cuh"


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


#define GROUP_REDUCE_POST_PROCESS_NUM_SMS 10
#define GROUP_REDUCE_POST_PROCESS_BLOCK_SIZE 1024


namespace torch::cuda::nccl {

    __device__ size_t binary_search_split(
        const int64_t* d_cu_split_size_list,
        size_t start,
        size_t end,
        size_t idx,
        size_t stride0
    ) {
        size_t low = start, high = end;
        while (low < high) { // [low, high)
            size_t mid = low + (high - low) / 2;
            if (idx < d_cu_split_size_list[mid] * stride0) {
                high = mid; // [low, mid)
            } else {
                low = mid + 1; // [mid + 1, high)
            }
        }
        return low - 1; // low == high
    }

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
            for (size_t r = 0; r < num_repeats; ++r) {
                recv_reduce_data += *(repeated_recv_data_ptr + r * recv_split_size);
            }

            // write the reduced data back to recv_buffer
            *recv_data_ptr = recv_reduce_data;
        }
    }

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

        int64_t repeat_dim_size = 0;
        for (int64_t output_split_idx = 0; output_split_idx < num_splits; ++output_split_idx) {
            auto num_repeats = src_indices_list[output_split_idx].size();
            auto split_size = output_split_size_list[output_split_idx];
            auto repeat_split_size = split_size * num_repeats;

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
        size_t num_splits
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
        dim3 gridDims(GROUP_REDUCE_POST_PROCESS_NUM_SMS); // we don't want the post-process kernel occupies too many SMs
        dim3 blockDims(GROUP_REDUCE_POST_PROCESS_BLOCK_SIZE);

        AT_DISPATCH_ALL_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, /* add float16/bfloat16 to dispatch types */
            type,
            "group_reduce_nccl_post_process_kernel",
            [&] {
            group_reduce_nccl_post_process_kernel<scalar_t> /* auto-deduced `scalar_t` by the macro */
                /** NOTE: the post-process kernel is supposed to run 
                 * on the same stream as the group reduce kernel, i.e. nccl stream
                 */
                <<<gridDims, blockDims, 0, stream.stream()>>>(
                    (scalar_t*) recv_buffer,
                    (scalar_t*) repeated_recv_buffer,
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

    }

} // namespace torch::cuda::nccl
