#pragma once

#ifndef USE_C10D_NCCL
#define USE_C10D_NCCL
#endif

#include <torch/python.h>

#include <torch/csrc/Stream.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/cuda/nccl.h>

/** NOTE: in this header file, pytorch defines a lot of type_caster 
 * to let pybind automatically convert between c++ and python types,
 * including torch.deivce, torch.Stream,
 * however, it seems to be local symbol in libtorch_python.so
 */
// #include <torch/csrc/utils/pybind.h>

#include <pybind11/chrono.h>

namespace c10d {

class TORCH_API ExtProcessGroupNCCL : public ProcessGroupNCCL {
public:
    // constructor
    ExtProcessGroupNCCL(
        c10::intrusive_ptr<Store> store,
        int rank,
        int size);
    
    // destructor
    ~ExtProcessGroupNCCL() override;

    // get the nccl cuda stream w.r.t. collective comm
    at::cuda::CUDAStream& getNCCLStream();

    // get the torch nccl comm
    /** NOTE: this api is valid but neither used nor binded for now,
     * 
     * since c10d::NCCLComm is a local symbol, the detailed debugging process is recorded as follows:
     * when using the member functions defined in NCCLComm as below:
     *      ncclComm_t ncclComm = torchNCCLComm->getNcclComm();
     *      ncclUniqueId nccUID = torchNCCLComm->getNcclId();
     * 
     * I run into an issue: undefined reference to `c10d::NCCLComm::getNcclComm()' 
     * later I've found out that all the member functions including`c10d::NCCLComm::getNcclComm()'
     * are local symbols that only visible inside the shared library `libtorch_cuda.so`,
     * 
     * with my own command as below:
     * nm /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so > libtorch_cuda.log
     * and the relevant output looks like:
     *      0000000000c98a40 t _ZN4c10d8NCCLComm11getNcclCommEv
     *      0000000000908e96 t _ZN4c10d8NCCLComm11getNcclCommEv.cold
     * 
     * thus we have no direct access to NCCLComm
     */
    std::shared_ptr<c10d::NCCLComm> getTorchNCCLComm();

    // get the nccl comm ptr w.r.t the current device
    /** NOTE: this API is only provided in the main branch of torch >= v-2.7.1
     * which can be a side way to get the nccl comm (ptr) w/o through torch nccl comm
    */
    // int64_t getNCCLCommPtr();

    int getDeviceID() const { return at::cuda::current_device(); }

    at::Device getDevice() const { return at::Device(at::kCUDA, getDeviceID()); }

    std::string getDeviceKey() const { return std::to_string(getDeviceID()); }

    // overrided collective interfaces
    c10::intrusive_ptr<Work> _allgather_base(
        at::Tensor& outputbuffer,
        at::Tensor& inputbuffer,
        const AllgatherOptions& opts = AllgatherOptions()) override;

    c10::intrusive_ptr<Work> alltoall_base(
        at::Tensor& outputTensor,
        at::Tensor& inputTensor,
        std::vector<int64_t>& outputSplitSizes,
        std::vector<int64_t>& inputSplitSizes,
        const AllToAllOptions& opts = AllToAllOptions()) override;

    // new collective interfaces
    c10::intrusive_ptr<Work> _dummy_allgather_base(
        at::Tensor& outputbuffer,
        at::Tensor& inputbuffer,
        const AllgatherOptions& opts = AllgatherOptions());

    // functor to call torch::cuda::nccl::all2all_single_equal_split
    struct All2AllSingleEqualSplitFunctor {
        int groupSize_;
        explicit All2AllSingleEqualSplitFunctor(int size) : groupSize_(size) {}
    
        ncclResult_t operator()(
            at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream
        ) const {
            torch::cuda::nccl::all2all_single_equal_split(
                input, output, groupSize_, comm, stream);
            return ncclSuccess;
        }
    };

    // functor to call torch::cuda::nccl::all2all_single_unequal_split
    struct All2AllSingleUnequalSplitFunctor {
        int groupSize_;
        std::vector<int64_t>& outputSplitSizes;
        std::vector<int64_t>& inputSplitSizes;
        explicit All2AllSingleUnequalSplitFunctor(
            int size, 
            std::vector<int64_t>& outputSplitSizes,
            std::vector<int64_t>& inputSplitSizes
        ) : groupSize_(size), outputSplitSizes(outputSplitSizes), inputSplitSizes(inputSplitSizes) {}

        ncclResult_t operator()(
            at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream
        ) const {
            std::vector<size_t> send_lengths(groupSize_);
            std::vector<size_t> recv_lengths(groupSize_);
            std::vector<size_t> send_offsets(groupSize_);
            std::vector<size_t> recv_offsets(groupSize_);
            c10d::computeLengthsAndOffsets(
                inputSplitSizes, input, &send_lengths, &send_offsets);
            c10d::computeLengthsAndOffsets(
                outputSplitSizes, output, &recv_lengths, &recv_offsets);
            // See [Sync Streams].
            torch::cuda::nccl::all2all_single_unequal_split(
                input.data_ptr(),
                send_lengths.data(),
                send_offsets.data(),
                output.data_ptr(),
                recv_lengths.data(),
                recv_offsets.data(),
                input.element_size(),
                input.scalar_type(),
                comm,
                stream);
            return ncclSuccess;
        }
    };

    // factory method to create an extended nccl process group
    static c10::intrusive_ptr<Backend> createExtProcessGroupNCCL(
        const c10::intrusive_ptr<::c10d::Store>& store,
        int rank,
        int size,
        const std::chrono::duration<float>& /* unused */
    );

    /** NOTE: this is a static method tagged with `__attribute__((constructor)`,
     * then it will be automatically called in class method `Backend.register_backend` in `distributed_c10d.py`
     * and this extended nccl process group backend object will be instantiated later using
     *  1. `backend_class = creator_fn(backend_prefix_store, group_rank, group_size, timeout)`, if `extended_api=False`
     *  2. `backend_class = creator_fn(dist_backend_opts, backend_options)`, otherwise
     * and the process group object will also register it using `pg._register_backend`
     */
    static void ExtProcessGroupNCCLConstructor() __attribute__((constructor)) {
        // import torch.distributed python module
        py::object module = py::module::import("torch.distributed");
        // access the register_backend method of the class Backend
        py::object register_backend = module.attr("Backend").attr("register_backend");
        // register using the factory method to create an extended nccl process group
        register_backend(
            "ext_nccl_backend", // backend name
            py::cpp_function(createExtProcessGroupNCCL), // func
            false, // extended_api: bool, set to false to not using backend options
            "cuda" // supported devices: Optional[Union[str, List[str]]] = None, set to only cuda
        );
    }
protected:
};
} // namespace c10d
