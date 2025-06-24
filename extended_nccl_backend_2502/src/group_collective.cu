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

namespace torch::cuda::nccl {

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
        at::cuda::CUDAStream& stream
    ) {
        auto num_input_splits = input_split_size_list.size();
        auto num_output_splits = output_split_size_list.size();

        auto nccl_data_type = to_nccl_data_type(type);
        auto nccl_comm = to_nccl_comm(comm);

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

        // post-process reduce from repeated_recv_buffer to recv_buffer
    }

} // namespace torch::cuda::nccl
