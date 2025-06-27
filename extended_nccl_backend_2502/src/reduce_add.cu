#include "launch_template.h"

// template void group_reduce_nccl_post_process_cute_kernel<float, 64>(
//     T_out* ptr_O,
//     T_out* ptr_R,

//     int64_t seqlen,
//     int64_t hidden_size,
//     int64_t seqlen_r,
//     int64_t num_splits,

//     const int64_t* cu_split_size_o,
//     const int64_t* split_size_list,
//     const int64_t* cu_split_size_r,
//     const int64_t* num_repeats_list,
//     cudaStream_t stream);
// template void group_reduce_nccl_post_process_cute_kernel<float, 128>(
//     T_out* ptr_O,
//     T_out* ptr_R,

//     int64_t seqlen,
//     int64_t hidden_size,
//     int64_t seqlen_r,
//     int64_t num_splits,

//     const int64_t* cu_split_size_o,
//     const int64_t* split_size_list,
//     const int64_t* cu_split_size_r,
//     const int64_t* num_repeats_list,
//     cudaStream_t stream);
// template void group_reduce_nccl_post_process_cute_kernel<float, 192>(Flash_fwd_params &params, cudaStream_t stream);

// template void group_reduce_nccl_post_process_cute_kernel<cutlass::bfloat16_t, 64>(Flash_fwd_params &params, cudaStream_t stream);
template void group_reduce_nccl_post_process_cute_kernel<cutlass::bfloat16_t, 128>(
    cutlass::bfloat16_t* ptr_O,
    cutlass::bfloat16_t* ptr_R,

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
);
// template void group_reduce_nccl_post_process_cute_kernel<cutlass::bfloat16_t, 192>(Flash_fwd_params &params, cudaStream_t stream);

// template void group_reduce_nccl_post_process_cute_kernel<cutlass::half_t, 64>(Flash_fwd_params &params, cudaStream_t stream);
// template void group_reduce_nccl_post_process_cute_kernel<cutlass::half_t, 128>(Flash_fwd_params &params, cudaStream_t stream);
// template void group_reduce_nccl_post_process_cute_kernel<cutlass::half_t, 192>(Flash_fwd_params &params, cudaStream_t stream);