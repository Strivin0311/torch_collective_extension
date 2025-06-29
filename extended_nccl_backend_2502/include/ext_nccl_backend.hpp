#pragma once

#ifndef USE_C10D_NCCL
#define USE_C10D_NCCL
#endif

#include <torch/python.h>

#include <ATen/cuda/CUDAGraph.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <torch/csrc/cuda/nccl.h>
#include <c10/util/WaitCounter.h>


/** NOTE: in this header file, pytorch defines a lot of type_caster 
 * to let pybind automatically convert between c++ and python types,
 * including torch.deivce, torch.Stream,
 * however, it seems to be local symbol in libtorch_python.so
 */
// #include <torch/csrc/utils/pybind.h>

#include <pybind11/chrono.h>

#include "ext_nccl_comm.hpp"
#include "group_collective.cuh"

namespace c10d {

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

struct GroupCastOptions {
    std::chrono::milliseconds timeout = kUnsetTimeout;
    bool asyncOp = true;
};

struct GroupReduceOptions {
    std::chrono::milliseconds timeout = kUnsetTimeout;
    bool asyncOp = true;
};


class TORCH_API ExtProcessGroupNCCL : public ProcessGroupNCCL {
public:
    // copied from WorkNCCL in torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp
    // since we need a work class to set ExtProcessGroupNCCL to be its friend class
    // REVIEW: is it better or worse to inherit from WorkNCCL?
    class ExtWorkNCCL : public Work, public std::enable_shared_from_this<ExtWorkNCCL> {
    public:
        friend struct WorkInfo;
    
        // Constructor takes a list of CUDA devices
        ExtWorkNCCL(
            std::string pgUID,
            std::string pgDesc,
            at::Device& device,
            int rank,
            OpType opType,
            uint64_t seq,
            bool isP2P = false,
            const char* profilingTitle = nullptr,
            const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt,
            bool desyncDebug = false,
            bool enableTiming = false,
            bool cudaEventCacheEnabled = false,
            DebugLevel distDebugLevel = DebugLevel::Off);
        // Copy constructor doing partial copy without outputs_. Cleanup thread
        // monitors and removes finished works. However it will deadlock when
        // destructs outputs_ tensors who are view tensors in autograd graph.
        ExtWorkNCCL(const ExtWorkNCCL& w);
    
        ~ExtWorkNCCL() override;
    
        // Checks if the NCCL kernel has started to execute.
        bool isStarted();
    
        // Checks if request has completed. In this specific case of NCCL, it checks
        // if the NCCL operation has completed on the GPU in its own NCCL stream.
        // Non-blocking operation.
        bool isCompleted() override;
    
        bool isSuccess() const override;
    
        // Same as calling synchronize() for NCCL work if timeout is not set.
        // Otherwise, it will block the CPU thread until the NCCL work is completed
        // or timed out. If timeout, exception will be thrown.
        bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
    
        void abort() override;
    
        // Let current stream wait on the completion of the NCCL work
        // Throws on exceptions.
        void synchronize() override;
    
        // Synchronize streams by blocking each on the NCCL stream
        void synchronizeStream();
    
        // Helper function to handle exception (throw if needed).
        void handleException(ErrorHandlingMode asyncErrorHandling);
    
        // Helper function that checks if the NCCL kernels have finished
        // execution on the GPUs
        bool finishedGPUExecution();
    
        // Get a Future object that will be marked as completed internally.
        c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;
    
        // Get a Future result of each work (e.g. success, different error types).
        // instead of the tensor output.
        c10::intrusive_ptr<c10::ivalue::Future> getFutureResult() override;
    
        float getDuration() const override;
    
        uint64_t getSequencenumber() const override;
    
        const std::string& logPrefix() const;
    
        // Helper function that sets an exception_ptr on the ExtWorkNCCL object.
        void setException(std::exception_ptr exception_ptr);
    
        // Helper function that returns True if the ExtWorkNCCL object has timed out
        // and False otherwise.
        // In case of timeout, set exception on the ExtWorkNCCL object.
        bool checkTimeout(
            std::optional<std::chrono::milliseconds> timeout = std::nullopt);
    
        // Print the traceback of the collective at call time
        void printTraceback() const;
    
        std::vector<at::Tensor> result() override;
    
    protected:
        // The process group unique id
        std::string pgUID_;
    
        // The process group description
        std::string pgDesc_;
    
        // The cached list of CUDA devices to operate on
        at::Device device_;
    
        // The start CUDA event of NCCL operator tracking this work item. These
        // start CUDA events are needed by desync debugging if enabled.
        std::shared_ptr<at::cuda::CUDAEvent> ncclStartEvent_;
    
        // The end CUDA event of NCCL operator tracking this work item.
        std::shared_ptr<at::cuda::CUDAEvent> ncclEndEvent_;

        // The ext NCCL communicator used for this work item.
        std::shared_ptr<ExtNCCLComm> extNcclComm_;
    
        // whether this work is a barrier op
        bool isBarrierOp_{false};
    
        // Clone of blockingWait_ from ProcessGroupNCCL.
        bool blockingWait_{false};
    
        // Clone of avoidRecordStreams_ from ProcessGroupNCCL.
        bool avoidRecordStreams_{false};
    
        // Clone of opTimeout_ from ProcessGroupNCCL.
        std::chrono::milliseconds opTimeout_{};
    
        // Ephemeral timeouts are owned by exactly one work,
        // and reset after that work completes.
        // There may be more than one ephemeral timeout active at the same time,
        // and this variable is used to track the ownership of ephemeral timeout.
        std::chrono::milliseconds ownedEphermeralTimeout_ =
            std::chrono::milliseconds(0);
    
        // Time point representing when the work started.
        std::chrono::time_point<std::chrono::steady_clock> workStartTime_;
    
        // Record the sequential number of collective or p2p.
        uint64_t seq_;
        bool isP2P_;
    
        // Indicates if the nccl start event has been updated to the store trace.
        // This will be used by desync debug.
        bool startTraceUpdated_{false};
    
        // Record collective sizes for debug. We only record the size on the first
        // device as multi-device per process is deprecated
        size_t numelIn_ = -1;
        size_t numelOut_ = -1;
    
        // Wrapper method for the static checkForNCCLErrors which can be overridden
        // for tests.
        virtual std::exception_ptr checkForNCCLErrors();
    
        friend std::ostream& operator<<(
            std::ostream& output,
            const ExtWorkNCCL& ExtWorkNCCL);
    
    private:
        // Checks for NCCL errors and sets an appropriate exception_ptr.
        void checkAndSetException();
    
        // Just checks whether GPU execution has started, without modifying
        // exception_ptr.
        bool startedGPUExecutionInternal() const;
    
        // Just checks whether GPU execution has completed, without modifying
        // exception_ptr.
        bool finishedGPUExecutionInternal() const;
    
        // Reference to the store so that we can write aborted communicators
        // to the store.
        c10::intrusive_ptr<Store> store_;
    
        // Store a reference to NCCL collective's outputs, used by result and to
        // give a more descriptive message when representing the Work as a string.
        std::shared_ptr<std::vector<at::Tensor>> outputs_;
    
        // TORCH_NCCL_AVOID_RECORD_STREAMS implementation helper.
        // Stores references to participating non-output tensors (ie inputs,
        // flattened intermediates).
        // We'll clear this list in synchronizeStream, just after user-facing
        // stream(s) are synced with the nccl work stream(s).
        // By keeping these refs (as well as outputs_) alive until after the
        // collective's work rejoins the user-facing streams, we achieve
        // caching allocator safety without any recordStream calls.
        // For in-place collectives, some refs stashed here may alias outputs_,
        // but that doesn't do any harm.
        std::shared_ptr<std::vector<at::Tensor>> stashed_for_allocator_safety_;
    
        // The future returned by getFuture.
        c10::intrusive_ptr<at::ivalue::Future> future_;
    
        // the future result (e.g., success or failure) of the work
        c10::intrusive_ptr<at::ivalue::Future> futureWorkResult_;
    
        bool timingEnabled_;
        // unique id used to tell the trace buffer that this
        // work has completed
        std::optional<uint64_t> trace_id_;
        DebugLevel distDebugLevel_;
        friend class ExtProcessGroupNCCL;
    };

    // constructor
    ExtProcessGroupNCCL(
        c10::intrusive_ptr<Store> store,
        int rank,
        int size);
    
    // destructor
    ~ExtProcessGroupNCCL() override;

    // Destroy (shutdown) this backend -- normal exit.
    void shutdown(); // override to append ext shutdown

    void ext_shutdown(); // to destroy the extended nccl comm objects besides the original nccl comm

    void startCoalescing() override;

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

    c10::intrusive_ptr<Work> alltoall(
        std::vector<at::Tensor>& outputTensors,
        std::vector<at::Tensor>& inputTensors,
        const AllToAllOptions& opts = AllToAllOptions()) override;

    // new collective interfaces
    c10::intrusive_ptr<Work> _dummy_allgather_base(
        at::Tensor& outputbuffer,
        at::Tensor& inputbuffer,
        const AllgatherOptions& opts = AllgatherOptions());

    c10::intrusive_ptr<Work> extended_alltoall_base(
        at::Tensor& outputTensor,
        at::Tensor& inputTensor,
        std::vector<int64_t>& outputSplitSizes,
        std::vector<int64_t>& inputSplitSizes,
        const AllToAllOptions& opts = AllToAllOptions());

    c10::intrusive_ptr<Work> group_cast(
        at::Tensor& outputTensor,
        at::Tensor& inputTensor,
        std::vector<int64_t>& inputSplitSizeList,
        std::vector<int64_t>& outputSplitSizeList,
        std::vector<std::vector<int64_t>>& dstIndicesList,
        std::vector<int64_t>& srcIndexList);
        // const GroupCastOptions& opts = GroupCastOptions());

    c10::intrusive_ptr<Work> group_reduce(
        at::Tensor& outputTensor,
        at::Tensor& inputTensor,
        std::vector<int64_t>& inputSplitSizeList,
        std::vector<int64_t>& outputSplitSizeList,
        std::vector<int64_t>& dstIndexList,
        std::vector<std::vector<int64_t>>& srcIndicesList);
        // const GroupCastOptions& opts = GroupReduceOptions());

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
    int globalRankStart_, globalRankStride_;

    // Vector to Store ExtWorkNCCL pointers
    std::list<ExtProcessGroupNCCL::ExtWorkNCCL> extWorkMetaList_;

    std::list<ExtProcessGroupNCCL::ExtWorkNCCL> completedExtWorkList_;

    std::unordered_map<std::string, std::shared_ptr<ExtNCCLComm>> devExtNCCLCommMap_;

    // The NCCL communicators currently in process of being initialized.
    std::unordered_map<std::string, std::shared_ptr<ExtNCCLComm>> inInitializationExtCommMap_;


    template <typename Fn>
    c10::intrusive_ptr<Work> ext_collective(
        at::Tensor& input,
        at::Tensor& output,
        Fn fn,
        OpType opType,
        const char* profilingTitle = nullptr,
        bool avoidRecordStreams = false,
        bool nanCheck = true);

    template <typename Fn, typename PreProcess, typename PostProcess>
    c10::intrusive_ptr<Work> ext_collective(
        at::Tensor& input,
        at::Tensor& output,
        Fn fn,
        PreProcess pre,
        PostProcess post,
        OpType opType,
        const char* profilingTitle = nullptr,
        bool avoidRecordStreams = false,
        bool nanCheck = true);

    template <typename Fn, typename PreProcess, typename PostProcess>
    c10::intrusive_ptr<Work> ext_collective(
        std::vector<at::Tensor>& inputs,
        std::vector<at::Tensor>& outputs,
        Fn fn,
        PreProcess pre,
        PostProcess post,
        OpType opType,
        const char* profilingTitle = nullptr,
        bool avoidRecordStreams = false,
        bool nanCheck = true);

    void assignTimeoutToExtWork(
        const c10::intrusive_ptr<ExtProcessGroupNCCL::ExtWorkNCCL>& work,
        const c10::intrusive_ptr<Options>& option);

    // Checks for NCCL errors on each of the communicators and returns an
    // appropriate exception_ptr (nullptr if no errors).
    static std::exception_ptr checkForNCCLErrorsInternal(
        std::shared_ptr<ExtNCCLComm>& ncclComm);

    // Ensure thaht if record is True, the work obj will be enqueued via
    // workEnqueue
    virtual c10::intrusive_ptr<ExtProcessGroupNCCL::ExtWorkNCCL> initExtWork(
        at::Device& device,
        int rank,
        OpType opType,
        bool isP2P,
        const char* profilingTitle = nullptr,
        const std::vector<at::Tensor>& inputs = {},
        const std::vector<at::Tensor>& outputs = {},
        bool record = false);

    // Add ExtWork Pointer to workVector
    void extWorkEnqueue(const c10::intrusive_ptr<ExtProcessGroupNCCL::ExtWorkNCCL>&);

    // Helper that looks up the cached extended NCCL communicators only
    std::shared_ptr<ExtNCCLComm> getExtNCCLComm(const std::string& deviceKey);

    std::shared_ptr<ExtNCCLComm> initExtNCCLComm(
        const std::string& deviceKey,
        at::Device& device,
        OpType opType,
        int p2pRank = 0,
        bool isSendRecvSelf = false);

    std::string createExtLogPrefix() const;

    // Returns the global ranks of a PG.
    const std::vector<uint64_t>& extGroupRanks() const;
};

} // namespace c10d
