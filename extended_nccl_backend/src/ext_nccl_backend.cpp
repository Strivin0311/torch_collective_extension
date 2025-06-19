#include "../include/ext_nccl_backend.hpp"

#ifndef USE_C10D_NCCL
#define USE_C10D_NCCL
#endif


/** NOTE: copied from torch/csrc/distributed/c10d/init.cpp
 * to pybind-define a gil-safe destructor for this module
 */
namespace {
// Wrapper to ensure GIL is released before destructing ProcessGroupGloo
// TODO: move this somewhere more generally useful
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_{};

 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) noexcept = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
      default;
  IntrusivePtrNoGilDestructor& operator=(
      IntrusivePtrNoGilDestructor&&) noexcept = default;
  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // This ctor is very important; see
  // https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      // NOLINTNEXTLINE(bugprone-exception-escape)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  [[nodiscard]] T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return impl_;
  }
};

} // anonymous namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true)

template <typename T>
using intrusive_ptr_no_gil_destructor_class_ =
  py::class_<T, IntrusivePtrNoGilDestructor<T>>;


namespace c10d {

namespace {
  // Check validity of tensor
  void check_gpu_single_tensor(
    const at::Tensor& tensor,
    const bool p2p = false // whether operation is a P2P operation
  ) {
    if (!tensor.is_cuda() || tensor.is_sparse()) {
      C10_THROW_ERROR(ValueError, "Tensors must be CUDA and dense");
    }
    // Skip the following requirements for P2P operations
    if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
      if (p2p) {
        TORCH_WARN_ONCE(
            "Detected non-contiguous tensor in P2P operations. It is user "
            "responsibility to guarantee that source and destination tensors have "
            "the same contiguity format.");
      } else {
        C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
      }
    }
  }
} // anonymous namespace

// constructor
ExtProcessGroupNCCL::ExtProcessGroupNCCL(
  c10::intrusive_ptr<c10d::Store> store,
  int rank,
  int size) : ProcessGroupNCCL(store, rank, size) {
    int globalRankStart_, globalRankStride_;
    if (options_->global_ranks_in_group.empty()) {
      globalRankStart_ = 0;
    } else {
      globalRankStart_ = options_->global_ranks_in_group[0];
    }
  
    if (options_->global_ranks_in_group.empty()) {
      globalRankStride_ = 1;
    } else if (options_->global_ranks_in_group.size() == 1) {
      globalRankStride_ = 0;
    } else {
        bool ranksAreStrided = true;
        auto startRank = options_->global_ranks_in_group[0];
        auto stride =
            options_->global_ranks_in_group[1] - options_->global_ranks_in_group[0];
        for (std::vector<uint64_t>::size_type i = 0;
            i < options_->global_ranks_in_group.size();
            i++) {
          if (options_->global_ranks_in_group[i] != startRank + i * stride) {
            ranksAreStrided = false;
            break;
          }
        }
    
        if (ranksAreStrided) {
          globalRankStride_ = options_->global_ranks_in_group[1] -
              options_->global_ranks_in_group[0];
        } else {
          globalRankStride_ = -1;
        }
    }
}

// destructor
ExtProcessGroupNCCL::~ExtProcessGroupNCCL() = default;

// get the nccl cuda stream w.r.t. collective comm
at::cuda::CUDAStream& ExtProcessGroupNCCL::getNCCLStream() {
  return ncclStreams_.at(getDeviceKey());
}

// get the torch nccl comm
std::shared_ptr<c10d::NCCLComm> ExtProcessGroupNCCL::getTorchNCCLComm() {
  return devNCCLCommMap_.at(getDeviceKey());
}

// get the nccl comm ptr
// int64_t ExtProcessGroupNCCL::getNCCLCommPtr() {
//     return c10d::ProcessGroupNCCL::getCommPtr();
// }


template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ExtProcessGroupNCCL::ext_collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType,
    const char* profilingTitle,
    bool avoidRecordStreams,
    bool nanCheck) {
  // Environment setting by the user may add onto collective call's option
  avoidRecordStreams |= avoidRecordStreams_;
  nanCheck &= enableNanCheck_;

  auto device = getDevice(inputs[0]);
  // Guard must be created before `currentStreamCaptureStatusMayInitCtx`;
  // otherwise, extra CUDA context could be created on device 0.
  at::cuda::OptionalCUDAGuard gpuGuard(device);

  c10::cuda::CaptureStatus capture_status =
      c10::cuda::currentStreamCaptureStatusMayInitCtx();
  errorIfCapturingNonCapturableNCCL(capture_status);

  // Bump collective counter
  if (!coalescing_state_) {
    seqCollective_++;
  }
  op_id_++;

  const auto key = getKeyFromDevice(device);
  std::shared_ptr<NCCLComm> ncclComm = getNCCLComm(key);
  if (ncclComm == nullptr) {
    ncclComm = initNCCLComm(key, device, opType);
  }

  if (coalescing_state_ & CoalActive) {
    if ((coalescing_state_ & CoalColl) == 0) {
      // First op in coalesced operations
      seqCollective_++;
    }
    coalescing_state_ |= CoalColl;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = ncclComm;
    } else {
      TORCH_CHECK(coalescedComm_ == ncclComm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  // Used many times below, so we stash the unordered_map lookup
  auto ncclStream = ncclStreams_.at(key);

  // First let NCCL streams wait for input tensors allocation streams
  syncStream(device, ncclEvents_[key], ncclStream);

  bool enqueue =
      !coalescing_state_ && capture_status == c10::cuda::CaptureStatus::None;
  auto work = initWork(
      device, rank_, opType, false, profilingTitle, inputs, outputs, enqueue);

  // Store references to outputs to be used by WorkNCCL::result and operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  if (avoidRecordStreams) {
    work->stashed_for_allocator_safety_ =
        std::make_shared<std::vector<at::Tensor>>(inputs);
  }

  if (nanCheck) {
    at::cuda::CUDAStreamGuard ncclStreamGuard(ncclStream);
    for (const auto& input : inputs) {
      bool nan = isnan(input)._is_any_true().item<bool>();
      if (nan) {
        throw std::runtime_error("NaN check failed in collective()");
      }
    }
  }

  // Start event should only be recorded before the ncclGroupStart()
  if (work->timingEnabled_) {
    work->ncclStartEvent_->record(ncclStream);
  }

  pre(ncclStream, work);

  ncclComm_t comm = ncclComm->getNcclComm();

  // Both `inputs' and `outputs' are created on a worker stream and used in
  // different ncclStreams.  Hence, both must record the ncclStream to
  // prevent being freed before the collective finishes.
  //
  // We only record `inputs' here, and leave recording `outputs' to `fn' for
  // operations where `inputs' and `outputs' are not the same.
  //
  // See [Sync Streams].
  if (!avoidRecordStreams) {
    for (const auto& input : inputs) {
      if (!input.is_sparse()) {
        c10::cuda::CUDACachingAllocator::recordStream(
            input.storage().data_ptr(), ncclStream);
      } else {
        // for sparse input case record streams on both index and value
        // tensors
        c10::cuda::CUDACachingAllocator::recordStream(
            input.values().storage().data_ptr(), ncclStream);
        c10::cuda::CUDACachingAllocator::recordStream(
            input.indices().storage().data_ptr(), ncclStream);
      }
    }
  }

// Not all collectives have the same signature, e.g, all-reduce take in a Tensor
// as the input and output while all-to-all take in a vector of Tensors as input
// and output. Because we define the signature of the fn to take only single
// tensor as input and output, we need to do a hack to get the first element in
// the vector and pass it to fn.
// TODO: we should clean up this in future (by either entirely removing lambda's
// or removing input and output from lambda's signature).
#ifndef NCCL_HAS_COMM_NONBLOCKING
  C10D_NCCL_CHECK(
      fn(inputs[0], outputs[0], comm, ncclStream),
      ncclComm->getNcclCommFailureReason());
#else
  C10D_NCCL_CHECK_TIMEOUT(
      fn(inputs[0], outputs[0], comm, ncclStream),
      comm,
      ncclComm->getNcclCommFailureReason());
#endif

  post(ncclStream, work);

  // End event should only be recorded after the ncclGroupEnd()
  if (!coalescing_state_) {
    work->ncclEndEvent_->record(ncclStream);
  }
  work->ncclComm_ = ncclComm;

  {
    c10::cuda::CUDAMultiStreamGuard streamGuard(ncclStream);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);

    // Add a callback that runs profiling end callbacks. wrapCallback() in CUDA
    // future blocks the stream this callback runs on the corresponding
    // ncclEndEvents_ ensuring appropriate synchronization.
    if (work->recordFunctionEndCallback_) {
      work->future_->addCallback(
          [work](at::ivalue::Future& /* unused */) {
            work->recordFunctionEndCallback_();
          },
          // uses_future = false allows us to skip synchronization in
          // ivalue::Future, but is only valid as long as the lambda doesn't use
          // the "Future" argument.
          /*uses_future=*/false);
    }
    work->future_->markCompleted(at::IValue(*work->outputs_));
  }

  // Set appropriate work parameters.
  work->blockingWait_ = blockingWait_;
  work->avoidRecordStreams_ = avoidRecordStreams;
  work->store_ = store_;
  assignTimeoutToWork(work, options_);
  // Record size info for debug. We only record the size on the first device as
  // multi-device per process is deprecated
  work->numelIn_ = 0;
  work->numelOut_ = 0;
  for (const auto& input : inputs) {
    work->numelIn_ += input.numel();
  }
  for (const auto& output : outputs) {
    work->numelOut_ += output.numel();
  }

  // Notify graphs before we check the capture status preemptively
  at::cuda::CUDAGraph::inc_pending_event_queries();
  if (enqueue) {
    workEnqueue(work);
  } else {
    at::cuda::CUDAGraph::dec_pending_event_queries();
  }

  return work;
}


// overrided collective interfaces
c10::intrusive_ptr<Work> ExtProcessGroupNCCL::_allgather_base(
  at::Tensor& outputbuffer,
  at::Tensor& inputbuffer,
  const AllgatherOptions& opts
) { // fake override
  return ProcessGroupNCCL::_allgather_base(
      outputbuffer,
      inputbuffer,
      opts
  );
}

c10::intrusive_ptr<Work> ExtProcessGroupNCCL::alltoall_base(
  at::Tensor& outputTensor,
  at::Tensor& inputTensor,
  std::vector<int64_t>& outputSplitSizes,
  std::vector<int64_t>& inputSplitSizes,
  const AllToAllOptions& /* unused */
) {
  check_gpu_single_tensor(outputTensor);
  check_gpu_single_tensor(inputTensor);

  int globalRankStart_ = getDeviceID(), globalRankStride_ = 1;

  auto inputs = std::vector<at::Tensor>{inputTensor};
  auto outputs = std::vector<at::Tensor>{outputTensor};

  if (outputSplitSizes.empty() && inputSplitSizes.empty()) {
    RECORD_PARAM_COMMS_DATA(
        std::make_tuple(
            static_cast<int64_t>(seqCollective_) + 1,
            false), // seq + 1 to match collective
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_all", // collective name
        inputTensor.numel(), // inNelems
        outputTensor.numel(), // outNelems
        inputTensor.scalar_type(), // dType
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>(), // outSplitSizes
        globalRankStart_, // globalRankStart
        globalRankStride_, // globalRankStride
        this->getSize()); // worldSize

    // avoidRecordStreams_ note: collective() will stash inputTensors and
    // outputTensors.
    return ext_collective(
        inputs,
        outputs,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          // See [Sync Streams].
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          // torch::cuda::nccl::all2all_single_equal_split(
          //     input, output, this->getSize(), comm, stream);
          return ncclSuccess;
        },
        [](at::cuda::CUDAStream&,
          c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
        [](at::cuda::CUDAStream&,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
        OpType::ALLTOALL_BASE,
        "nccl:all_to_all");
  } else {
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);

    RECORD_PARAM_COMMS_DATA(
        std::make_tuple(
            static_cast<int64_t>(seqCollective_) + 1,
            false), // seq + 1 to match collective
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_allv", // collective name
        inputTensor.numel(), // inNelems
        outputTensor.numel(), // outNelems
        inputTensor.scalar_type(), // dType
        inputSplitSizes, // inSplitSizes
        outputSplitSizes, // outSplitSizes
        globalRankStart_, // globalRankStart
        globalRankStride_, // globalRankStride
        this->getSize()); // worldSize

    // avoidRecordStreams_ note: collective() will stash inputTensors and
    // outputTensors.
    return ext_collective(
        inputs,
        outputs,
        [&](at::Tensor& input,
            at::Tensor& output,
            ncclComm_t comm,
            at::cuda::CUDAStream& stream) {
          std::vector<size_t> send_lengths(size_);
          std::vector<size_t> recv_lengths(size_);
          std::vector<size_t> send_offsets(size_);
          std::vector<size_t> recv_offsets(size_);
          c10d::computeLengthsAndOffsets(
              inputSplitSizes, input, &send_lengths, &send_offsets);
          c10d::computeLengthsAndOffsets(
              outputSplitSizes, output, &recv_lengths, &recv_offsets);
          // See [Sync Streams].
          if (!avoidRecordStreams_) {
            c10::cuda::CUDACachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
          // torch::cuda::nccl::all2all_single_unequal_split(
          //     input.data_ptr(),
          //     send_lengths.data(),
          //     send_offsets.data(),
          //     output.data_ptr(),
          //     recv_lengths.data(),
          //     recv_offsets.data(),
          //     input.element_size(),
          //     input.scalar_type(),
          //     comm,
          //     stream);
          return ncclSuccess;
        },
        [](at::cuda::CUDAStream&,
          c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
        [](at::cuda::CUDAStream&,
            c10::intrusive_ptr<ProcessGroupNCCL::WorkNCCL>& work) {},
        OpType::ALLTOALL_BASE,
        "nccl:all_to_all");
  }
}


// new collective interfaces
c10::intrusive_ptr<Work> ExtProcessGroupNCCL::_dummy_allgather_base(
  at::Tensor& outputbuffer,
  at::Tensor& inputbuffer,
  const AllgatherOptions& opts
) {
  printf("This is a dummy _allgather_base that sets output buffer to zero\n");
  outputbuffer.zero_();

  auto inputs = std::vector<at::Tensor>{inputbuffer};
  auto outputs = std::vector<at::Tensor>{outputbuffer};

  auto device = getDevice();
  int rank = getDeviceID();

  auto work = initWork(
      device,
      rank,
      OpType::ALLGATHER,
      false, /*isP2P*/
      "_allgather_base_dummy",
      inputs,
      outputs,
      true /*record*/
  );

  return work;
}


// factory method to create an extended nccl process group
c10::intrusive_ptr<Backend> ExtProcessGroupNCCL::createExtProcessGroupNCCL(
    const c10::intrusive_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& /* unused */
  ) {
  // return c10::make_intrusive<ExtProcessGroupNCCL>(rank, size);
  return c10::make_intrusive<ExtProcessGroupNCCL>(store, rank, size);
}

/** NOTE: `TORCH_EXTENSION_NAME` is an env var
 * that will be automatically translated to the extention module name defined in setup.py
 * e.g. since this module is named `ext_nccl_backend`
 * thus in the python script, we can use this function (though no use for now) as follows:
 * import ext_nccl_backend; print(ext_nccl_backend.createExtProcessGroupNCCL)
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createExtProcessGroupNCCL", &ExtProcessGroupNCCL::createExtProcessGroupNCCL);

  auto torch_c10d = py::module::import("torch._C._distributed_c10d");
  auto processGroupNCCL = torch_c10d.attr("ProcessGroupNCCL"); // inherit from ProcessGroupNCCL
  auto module = py::handle(m).cast<py::module>();

  auto extProcessGroupNCCL = 
      intrusive_ptr_no_gil_destructor_class_<ExtProcessGroupNCCL>(
          module, "ExtProcessGroupNCCL", processGroupNCCL)
      .def(
          py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                     int rank,
                     int size) {
            // copied from torch/csrc/distributed/c10d/init.cpp
            // gil_scoped_release is not safe as a call_guard for constructor
            // see: https://github.com/pybind/pybind11/issues/5473
            py::gil_scoped_release nogil;
            return c10::make_intrusive<ExtProcessGroupNCCL>(
                store, rank, size);
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          "Constructor to create ExtProcessGroupNCCL instance"
      )
      .def(
        "_dummy_allgather_base",
        &ExtProcessGroupNCCL::_dummy_allgather_base,
        py::arg("output"),
        py::arg("input"),
        py::arg("opts")
      )
      .def_property_readonly(
        "nccl_stream",
        [](ExtProcessGroupNCCL& self) -> py::object {
          /** NOTE: here we do some hacky thing to get the nccl cuda stream in python-end
           * 
           * We first list the limitations as follows:
           *    1. at::cuda::CUDAStream` is not registered by pytorch in python-end,
           *      and we need to unwrap it to c10::Stream
           *    2. it is not c10::Stream, but THPStream, that is directly linked to torch.cuda.Stream,
           *      thus we need to convert a c10::Stream to THPStream
           *    3. although pytorch gives a `THPStream_Wrap` function in `torch/csrc/Stream.h`
           *      as well as a pybind type_cast function in `torch/csrc/utils/pybind.h`,
           *      THPStream_Wrap is a local symbol in /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_python.so
           *      thus we cannot directly access it
           * 
           * As a result, we give up the following code:
           *    c10::Stream c10_stream = self.getNCCLStream().unwrap();
           *    return py::reinterpret_steal<py::object>(THPStream_Wrap(c10_stream));
           * 
           * Therefore, we directly access the torch.cuda.Stream module 
           * and initialize a pybind object with the internal cuda stream ptr as kwargs
           */

          thread_local py::object cached_nccl_stream = py::none();
          if (!cached_nccl_stream.is_none()) { // already cached
              return cached_nccl_stream;
          }

          /* everything is ok, only the stream id is not identical, but seems no problem  */
          at::cuda::CUDAStream cuda_stream = self.getNCCLStream();
          auto torch = py::module::import("torch");
          auto torch_cuda_stream_class = torch.attr("cuda").attr("Stream");
          py::kwargs kwargs;
          kwargs["stream_ptr"] = py::cast(reinterpret_cast<uintptr_t>(cuda_stream.stream()));
          cached_nccl_stream = torch_cuda_stream_class(**kwargs);
          return cached_nccl_stream;

          /* valid but only create a torch.Stream, and the stream ptr seems to be wrong */
          // auto cuda_stream_base = torch.attr("_C").attr("_CudaStreamBase");
          // return cuda_stream_base(
          //     py::arg("priority") = 0,
          //     py::arg("stream_id") = 0,
          //     py::arg("device_index") = cuda_stream.device_index(),
          //     py::arg("stream_ptr") = reinterpret_cast<uintptr_t>(cuda_stream.stream())
          // );

          /* invalid: RuntimeError: Expected stream_.device_type() == DeviceType::CUDA to be true, but got false */
          // auto kwargs = py::dict();
          // kwargs["device_index"] = py::cast(cuda_stream.device_index());
          // kwargs["stream_id"] = py::cast(cuda_stream.id());
          // kwargs["stream_ptr"] = py::cast(reinterpret_cast<uintptr_t>(cuda_stream.stream()));
          // return torch_cuda_stream_class(**kwargs);

          /* valid but stream ptr wrong */
          // return torch_cuda_stream_class(
          //     py::cast(cuda_stream.device_index()),
          //     py::cast(reinterpret_cast<uintptr_t>(cuda_stream.stream()))
          // );

          /* invalid */
          // return torch_cuda_stream_class(
          //     py::none(),                   // priority (ignored)
          //     py::none(),                   // stream_id (ignored)
          //     py::cast(cuda_stream.device_index()),
          //     py::cast(static_cast<int64_t>(c10::DeviceType::CUDA)),
          //     py::cast(reinterpret_cast<uintptr_t>(cuda_stream.stream()))
          // );
        },
        R"(Return the NCCL cuda stream w.r.t the current device)"
      )
      ;
}

} // namespace c10d
