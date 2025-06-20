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

static constexpr int ExtCoalActive = 0x01, ExtCoalColl = 0x02, ExtCoalP2P = 0x04;

constexpr const char* EXT_MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple. You are probably using multiple "
    "devices under one thread. The support for such usage has been deprecated. "
    "For details, please refer to "
    "https://pytorch.org/docs/stable/distributed.html#multi-gpu-collective-functions. "
    "ProcessGroupNCCL continues supporting multi-process and multi-thread modes.";

constexpr int64_t kExtSynchronizeBusyWaitMillis = 1;


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

  // Get a key string from device
  inline std::string getKeyFromDevice(at::Device& device) {
    return std::to_string(device.index());
  }

  inline void errorIfCapturingNonCapturableNCCL(c10::cuda::CaptureStatus status) {
    // parentheses avoid some compiler warnings
    static const uint64_t min_version =
        (((uint64_t)2) << 32) + (((uint64_t)9) << 16) + ((uint64_t)6);
    static const uint64_t cur_version = torch::cuda::nccl::version();
    if (cur_version < min_version) {
      TORCH_CHECK_WITH(
          NotImplementedError,
          status == c10::cuda::CaptureStatus::None,
          "Capturing NCCL collectives is only allowed with NCCL >= 2.9.6");
    }
  }

  // Returns exception's what() given an exception_ptr instance.
  std::string getExceptionMsgFromExceptionPtr(
    const std::exception_ptr& exceptionPtr) {
    TORCH_CHECK(exceptionPtr != nullptr);
    try {
      std::rethrow_exception(exceptionPtr);
    } catch (const std::exception& e) {
      return e.what();
    } catch (...) {
      return "Unknown exception type";
    }
  }

  void syncStream(
      at::Device& device,
      at::cuda::CUDAEvent& ncclEvent,
      at::cuda::CUDAStream& ncclStream) {
    ncclEvent.record(at::cuda::getCurrentCUDAStream(device.index()));
    ncclEvent.block(ncclStream);
  }

} // anonymous namespace


// Map from each communicator to its device index.
// This map is used when register/deregister cache segments from cache
// allocator. See design notes below:
// - Each segment should be registered only to the communicator on the
//   same device.
// - We cannot reuse devExtNCCLCommMap_ in each ProcessGroup because the key may be
//   ranks rather than device in point-to-point case.
// - This map has also to be maintained as global variable since the register
//   hooks are called outside the scope of any PG, thus we need traverse
//   communicators in all PGs.
static std::unordered_map<std::shared_ptr<ExtNCCLComm>, int> extNcclCommDevIdxMap;
static std::mutex extNcclCommDevIdxMapMutex;

/***********          For internal extended nccl work          ***********/


std::ostream& operator<<(
  std::ostream& output,
  const ExtProcessGroupNCCL::ExtWorkNCCL& workNCCL
) {
  std::string workInfo;
  workInfo = c10::str(
      "WorkNCCL(",
      "SeqNum=",
      workNCCL.seq_,
      ", OpType=",
      opTypeToString(workNCCL.opType_),
      ", NumelIn=",
      workNCCL.numelIn_,
      ", NumelOut=",
      workNCCL.numelOut_,
      ", Timeout(ms)=",
      workNCCL.opTimeout_.count(),
      ")");
  return output << workInfo;
}

// constructor
ExtProcessGroupNCCL::ExtWorkNCCL::ExtWorkNCCL(
  std::string pgUID,
  std::string pgDesc,
  at::Device& device,
  int rank,
  OpType opType,
  uint64_t seq,
  bool isP2P,
  const char* profilingTitle,
  const std::optional<std::vector<at::Tensor>>& inputs,
  bool desyncDebug,
  bool enableTiming,
  bool cudaEventCacheEnabled,
  DebugLevel distDebugLevel)
  : Work(rank, opType, profilingTitle, inputs),
    pgUID_(std::move(pgUID)),
    pgDesc_(std::move(pgDesc)),
    device_(device),
    workStartTime_(std::chrono::steady_clock::now()),
    seq_(seq),
    isP2P_(isP2P),
    timingEnabled_(enableTiming),
    distDebugLevel_(distDebugLevel) {
  // Creates the CUDA event wrappers
  // Note: The actual events are lazily created when first recorded to with
  // DEFAULT_FLAGS = cudaEventDisableTiming.
  if (cudaEventCacheEnabled) {
    ncclStartEvent_ = enableTiming
        ? ProcessGroupNCCL::CUDAEventCache::get(device.index())
              .create(enableTiming)
        : nullptr;
    ncclEndEvent_ = ProcessGroupNCCL::CUDAEventCache::get(device.index())
                        .create(enableTiming);
  } else {
    ncclStartEvent_ = enableTiming
        ? std::make_shared<at::cuda::CUDAEvent>(cudaEventDefault)
        : nullptr;
    ncclEndEvent_ = std::make_shared<at::cuda::CUDAEvent>(
        enableTiming ? cudaEventDefault : cudaEventDisableTiming);
  }
  futureWorkResult_ =
      c10::make_intrusive<at::ivalue::Future>(c10::AnyEnumType::get());
}

// copy constructor
ExtProcessGroupNCCL::ExtWorkNCCL::ExtWorkNCCL(const ExtWorkNCCL& w)
  : Work(w.rank_, w.opType_),
    std::enable_shared_from_this<ExtWorkNCCL>(w),
    pgUID_(w.pgUID_),
    pgDesc_(w.pgDesc_),
    device_(w.device_),
    ncclStartEvent_(w.ncclStartEvent_),
    ncclEndEvent_(w.ncclEndEvent_),
    extNcclComm_(w.extNcclComm_),
    blockingWait_(w.blockingWait_),
    opTimeout_(w.opTimeout_),
    ownedEphermeralTimeout_(w.ownedEphermeralTimeout_),
    workStartTime_(w.workStartTime_),
    seq_(w.seq_),
    isP2P_(w.isP2P_),
    startTraceUpdated_(w.startTraceUpdated_),
    numelIn_(w.numelIn_),
    numelOut_(w.numelOut_),
    store_(w.store_),
    futureWorkResult_(w.futureWorkResult_),
    timingEnabled_(w.timingEnabled_),
    trace_id_(w.trace_id_),
    distDebugLevel_(w.distDebugLevel_) {
  exception_ = w.exception_;
}

ExtProcessGroupNCCL::ExtWorkNCCL::~ExtWorkNCCL() = default;

bool ExtProcessGroupNCCL::ExtWorkNCCL::isCompleted() {
  if (!extNcclComm_->isAborted()) {
    checkAndSetException();
  }
  return exception() || finishedGPUExecutionInternal();
}

bool ExtProcessGroupNCCL::ExtWorkNCCL::isStarted() {
  if (!extNcclComm_->isAborted()) {
    checkAndSetException();
  }
  return exception() || startedGPUExecutionInternal();
}

bool ExtProcessGroupNCCL::ExtWorkNCCL::isSuccess() const {
  C10_THROW_ERROR(NotImplementedError, "ExtWorkNCCL::isSuccess() is deprecated");
}

void ExtProcessGroupNCCL::ExtWorkNCCL::checkAndSetException() {
  if (exception()) {
    // We already have an exception.
    return;
  }

  auto exception_ptr = checkForNCCLErrors();
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = exception_ptr;
  if (exception_) {
    LOG(ERROR) << logPrefix() << "Collective " << *this
              << " raised the following async exception: "
              << getExceptionMsgFromExceptionPtr(exception_);

    // Mark future result as ERROR
    if (futureWorkResult_ && !futureWorkResult_->completed()) {
      futureWorkResult_->markCompleted(
          at::IValue(static_cast<uint8_t>(WorkResult::COMM_ERROR)));
    }
  }
}

const std::string& ExtProcessGroupNCCL::ExtWorkNCCL::logPrefix() const {
  static std::string prefix = c10::str("[Rank ", rank_, "] ");
  return prefix;
}

void ExtProcessGroupNCCL::ExtWorkNCCL::setException(
  std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = std::move(exception_ptr);
}

// Helper that checks if the NCCL kernels are completed on the GPUs
bool ExtProcessGroupNCCL::ExtWorkNCCL::finishedGPUExecution() {
  checkAndSetException();
  return finishedGPUExecutionInternal();
}

bool ExtProcessGroupNCCL::ExtWorkNCCL::startedGPUExecutionInternal() const {
  // if timing is disabled we won't have allocated start events
  if (!timingEnabled_) {
    return false;
  }
  // Checking the work's corresponding CUDA event's status
  if (!ncclStartEvent_->query()) {
    return false;
  }
  return true;
}

bool ExtProcessGroupNCCL::ExtWorkNCCL::finishedGPUExecutionInternal() const {
  // Checking the work's corresponding CUDA event's status
  // It calls `cudaEventQuery` eventually. Although this seems to be a
  // non-blocking call, but we did notice hangs in the past. It can
  // hang if another thread is holding the CUDA global context lock. For
  // example, when doing a `cudaDeviceSynchronize` or even
  // `cudaStreamSynchronize`.
  if (!ncclEndEvent_->query()) {
    return false;
  }
  return true;
}

bool ExtProcessGroupNCCL::ExtWorkNCCL::checkTimeout(
  std::optional<std::chrono::milliseconds> timeout) {
  STATIC_SCOPED_WAIT_COUNTER(
      pytorch.wait_counter.ProcessGroupNCCL__checkTimeout);
  auto currentTimepoint = std::chrono::steady_clock::now();
  auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTimepoint - workStartTime_);
  auto workTimeout = timeout ? *timeout : opTimeout_;

  if (timeElapsed < workTimeout) {
    return false;
  }

  // Timed out

  std::string exceptionMsg = c10::str(
      logPrefix(),
      "Watchdog caught collective operation timeout: ",
      *this,
      " ran for ",
      timeElapsed.count(),
      " milliseconds before timing out.");

  LOG(ERROR) << exceptionMsg;

  std::exception_ptr exception_ptr =
      std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exceptionMsg));
  if (!exception()) {
    // if there is already an error, we don't override it
    setException(exception_ptr);
  }

  // Mark future result as TIMEOUT
  if (futureWorkResult_ && !futureWorkResult_->completed()) {
    futureWorkResult_->markCompleted(
        at::IValue(static_cast<uint8_t>(WorkResult::TIMEOUT)));
  }
  return true;
}

// Print the traceback of the collective at call time
void ExtProcessGroupNCCL::ExtWorkNCCL::printTraceback() const {
  // First step we get the corresponding record entry from FR, based on work's
  // trace_id_
  std::optional<FlightRecorder::Entry> entry =
      FlightRecorder::get()->getEntry(trace_id_);
  if (entry.has_value()) {
    auto entryVal = entry.value();
    // Get stack trace from FR entry, in string format
    // Note: `getTraceback` call below invokes `torch::symbolize`, which may
    // need to acquire the GIL. In order for watchdog to be block-free, we make
    // the call with std::async.
    auto future = std::async(
        std::launch::async, [&entryVal]() { return entryVal.getTraceback(); });
    // Wait for the future to complete or timeout
    auto status = future.wait_for(std::chrono::seconds(8));
    if (status == std::future_status::ready) {
      std::string tracebackStr = future.get();
      LOG(ERROR) << "Stack trace of the failed collective: \n" << tracebackStr;
    } // else, symbolizer probably timed out, we skip logging the stack trace.
  } else {
    LOG(ERROR)
        << "Stack trace of the failed collective not found, "
        << "potentially because FlightRecorder is disabled. "
        << "You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.";
  }
}

void ExtProcessGroupNCCL::ExtWorkNCCL::handleException(
  ErrorHandlingMode errorHandling) {
  if (exception_) {
    auto exceptionMsg = c10::str(
        "Some NCCL operations have failed or timed out. Due to the ",
        "asynchronous nature of CUDA kernels, subsequent GPU operations ",
        "might run on corrupted/incomplete data.");
    LOG(ERROR) << logPrefix() << exceptionMsg;
    C10_LOG_API_USAGE_ONCE("ProcessGroupNCCL.ExtWorkNCCL.handleException");

    auto logger = c10d::C10dLogger::getLogger();
    if (logger) {
      ::c10d::C10dLoggingData data;
      data.strings["work_nccl_exception"] =
          getExceptionMsgFromExceptionPtr(exception_);
      logger->log(data);
    }

    if (SHOULD_TEAR_DOWN(errorHandling)) {
      auto tearDownMsg = c10::str(
          "To avoid data inconsistency, we are taking the entire process down.");
      LOG(ERROR) << logPrefix() << tearDownMsg;
      std::rethrow_exception(exception_);
    }
  }
}

void ExtProcessGroupNCCL::ExtWorkNCCL::synchronize() {
  synchronizeStream();
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<
            ExtProcessGroupNCCL::ExtWorkNCCL>::unsafe_reclaim_from_nonowning(this));
  }
}

void ExtProcessGroupNCCL::ExtWorkNCCL::synchronizeStream() {
  auto currentStream = at::cuda::getCurrentCUDAStream(device_.index());
  // Block the current stream on the NCCL stream
  ncclEndEvent_->block(currentStream);

  if (avoidRecordStreams_) {
    stashed_for_allocator_safety_->clear();
  }
}

// Same as calling synchronize() when blockingWait_ is false
bool ExtProcessGroupNCCL::ExtWorkNCCL::wait(std::chrono::milliseconds timeout) {
  RECORD_PARAM_COMMS(
      std::make_tuple(static_cast<int64_t>(this->seq_), this->isP2P_), // seq
      std::make_tuple(pgUID_, pgDesc_), // PG name tuple
      rank_, // rank
      "wait", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1,
      -1,
      static_cast<int>(1)); // number of device?

  // synchronize() will block the current stream on the NCCL stream
  synchronize();

  // In case of blockingWait or a timeout value is specified by the user, we
  // block the CPU thread until the work is completed or timed out.
  if (blockingWait_ || timeout != kNoTimeout) {
    while (!isCompleted()) {
      bool timedOut = checkTimeout(
          timeout == kNoTimeout ? std::nullopt : std::make_optional(timeout));
      // Explicitly abort ncclComms here before throwing this timed out
      // exception to users.
      // If throwing timed out excepiton without aborting nccl communicators
      // here, it was observed that CUDA GPU will have 100% utilization and
      // can not run new events successfully.
      if (timedOut) {
        std::string exceptionMsg = c10::str(
            logPrefix(), "Work ", (*this), " timed out in blocking wait.");
        LOG(ERROR) << exceptionMsg;
        break;
      }
      // Yield
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kExtSynchronizeBusyWaitMillis));
    }
  } else if (isBarrierOp_ && !isCompleted()) {
    // For barrier wait when timeout is unspecified, we block the CPU thread on
    // current stream. This is to minimize the CPU barrier wait time in healthy
    // path
    auto currentStream = at::cuda::getCurrentCUDAStream(device_.index());
    // CUDAStream wrapper will correctly use a DeviceGuard here
    currentStream.synchronize();
  }

  // If exception is detected, throw it from the main CPU thread
  if (exception()) {
    // Abort NCCL communicators
    abort();
    // Throw exception (from main thread here)
    handleException(TearDown);
  }

  // TODO(kwen2501): this should be moved to c10d tests, to qualify a NCCL
  // upgrade. Once a NCCL version is qualified, this code should not be needed
  // at runtime.
  #ifdef PGNCCL_ENABLE_HASH
  if (distDebugLevel_ >= DebugLevel::Detail) {
    auto numel = getTensorsNumel(*outputs_);
    auto hashValue = hashTensors(*outputs_);
    PRINT_COLLECTIVE_HASH_SIGNATURE(
        "output", opTypeToString(opType_), numel, hashValue);
  }
  #endif
  // Always return true, because abort API is not implemented.
  return true;
  }

void ExtProcessGroupNCCL::ExtWorkNCCL::abort() {
  // dump before aborting for rcclexp
  #if defined(USE_ROCM) && defined(NCCL_COMM_DUMP)
  auto dumpMap = extNcclComm_->ncclCommDump();
  printNcclCommProxyTrace("ExtWorkNCCL::abort", dumpMap);
  #endif

  // Abort all communicators of this work
  extNcclComm_->abort();

  extNcclCommDevIdxMapMutex.lock();
  extNcclCommDevIdxMap.erase(extNcclComm_);
  extNcclCommDevIdxMapMutex.unlock();
}

std::exception_ptr ExtProcessGroupNCCL::ExtWorkNCCL::checkForNCCLErrors() {
  return checkForNCCLErrorsInternal(extNcclComm_);
}

std::vector<at::Tensor> ExtProcessGroupNCCL::ExtWorkNCCL::result() {
  return *outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future> ExtProcessGroupNCCL::ExtWorkNCCL::
    getFuture() {
  return future_;
}

c10::intrusive_ptr<c10::ivalue::Future> ExtProcessGroupNCCL::ExtWorkNCCL::
    getFutureResult() {
  return futureWorkResult_;
}

float ExtProcessGroupNCCL::ExtWorkNCCL::getDuration() const {
  TORCH_CHECK(timingEnabled_, "getDuration only works if timing was enabled");
  TORCH_CHECK(
      ncclStartEvent_,
      "getDuration only works if ncclStartEvents_ is populated, true if timing enabled");
  TORCH_CHECK(
      ncclEndEvent_,
      "getDuration only works if ncclEndEvents_ is populated, which should always be true");
  return ncclStartEvent_->elapsed_time(*ncclEndEvent_);
}

uint64_t ExtProcessGroupNCCL::ExtWorkNCCL::getSequencenumber() const {
  return seq_;
}


/***********          For extended nccl process group          ***********/

// constructor
ExtProcessGroupNCCL::ExtProcessGroupNCCL(
  c10::intrusive_ptr<c10d::Store> store,
  int rank,
  int size
) : ProcessGroupNCCL(store, rank, size) {
    // init global rank start and stride
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


std::string ExtProcessGroupNCCL::createExtLogPrefix() const {
  if (!pg_desc_.empty() && pg_desc_ != "undefined") {
    return c10::str(
        "[PG ID ",
        local_id_,
        " PG GUID ",
        pg_uid_,
        "(",
        pg_desc_,
        ") Rank ",
        rank_,
        "] ");
  }
  return c10::str(
      "[PG ID ", local_id_, " PG GUID ", pg_uid_, " Rank ", rank_, "] ");
}


void ExtProcessGroupNCCL::assignTimeoutToExtWork(
  const c10::intrusive_ptr<ExtProcessGroupNCCL::ExtWorkNCCL>& work,
  const c10::intrusive_ptr<ProcessGroupNCCL::Options>& option
) {
  std::chrono::milliseconds timeout = option->timeout;
  std::lock_guard<std::mutex> timeoutLock(mtxTimeoutExtension_);
  if (ephemeralTimeoutActive_.count() > 0) {
    timeout += ephemeralTimeoutActive_;
  }
  work->opTimeout_ = timeout;
  work->ownedEphermeralTimeout_ =
      ephemeralTimeoutActive_ - ephemeralTimeoutInflight_;
  ephemeralTimeoutInflight_ = ephemeralTimeoutActive_;
}


c10::intrusive_ptr<ExtProcessGroupNCCL::ExtWorkNCCL> ExtProcessGroupNCCL::initExtWork(
  at::Device& device,
  int rank,
  OpType opType,
  bool isP2P,
  const char* profilingTitle,
  const std::vector<at::Tensor>& inputs,
  const std::vector<at::Tensor>& outputs, // TODO(kwen2501): necessary?
  bool record
) {
  auto r = c10::make_intrusive<ExtProcessGroupNCCL::ExtWorkNCCL>(
      pg_uid_,
      pg_desc_,
      device,
      rank,
      opType,
      isP2P ? seqP2P_ : seqCollective_,
      isP2P,
      profilingTitle,
      profilingTitle != nullptr ? std::optional<std::vector<at::Tensor>>(inputs)
                                : std::nullopt,
      desyncDebug_,
      enableTiming_.load(),
      cudaEventCacheEnabled_.load(),
      dist_debug_level_);

  if (record) {
    bool isP2P = isP2POp(opType);
    // Ideally record every work that we enqueue, rather than every work we
    // create.
    // - at the time of this PR we do not currently enqueue every created work
    // - but it is unsafe to steal refs to start/end cuda events from Works that
    //   may go out of scope before flight recorder has retired them,
    //   so we must ensure that any work that is initialized via initExtWork will
    //   be enqueued
    // - initially, moved record() into workEnqueue(), but found that makes it
    //   hard to get access to profilingTitle,
    //   inputs, and outputs for metadata recording, and we don't want to attach
    //   these objects to the Work becuase it has implications for keeping those
    //   tensors alive longer and adds overhead when copying Work objects
    //   between threads
    r->trace_id_ = FlightRecorder::get()->record(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_),
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle ? profilingTitle : "",
        inputs,
        outputs,
        r->ncclStartEvent_.get(),
        r->ncclEndEvent_.get(),
        options_->timeout,
        pgStatus_,
        isP2P);
  }
  return r;
}


const std::vector<uint64_t>& ExtProcessGroupNCCL::extGroupRanks() const {
  if (options_->global_ranks_in_group.empty() && local_id_ == 0) {
    static std::vector<uint64_t> globalRanks(size_);
    std::iota(globalRanks.begin(), globalRanks.end(), 0);
    return globalRanks;
  }
  return options_->global_ranks_in_group;
}


std::shared_ptr<ExtNCCLComm> ExtProcessGroupNCCL::getExtNCCLComm(
  const std::string& deviceKey
) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (devExtNCCLCommMap_.find(deviceKey) != devExtNCCLCommMap_.end()) {
    // Reuse the cached communicator if there is one.
    return devExtNCCLCommMap_[deviceKey];
  }
  return nullptr;
}


std::shared_ptr<ExtNCCLComm> ExtProcessGroupNCCL::initExtNCCLComm(
  const std::string& deviceKey,
  at::Device& device,
  OpType opType,
  int p2pRank,
  bool isSendRecvSelf
) {
  // Sanity check
  if (deviceKey.empty()) {
    C10_THROW_ERROR(
        DistBackendError,
        "Not able to create/get the NCCL Communicator since "
        "the GPU devices are not known");
  }
  if (bound_device_id_) {
    if (*bound_device_id_ != device) {
      LOG(ERROR) << logPrefix_ << "Tensor found on device " << device
                << " but backend constrained to " << *bound_device_id_;
      C10_THROW_ERROR(
          DistBackendError,
          "Attempt to perform collective on tensor not on device passed to init_process_group");
    }
  }

  usedDeviceIdxs_.insert(device.index());

  // extended NCCL communicator not cached, create a new entry
  // std::shared_ptr<NCCLComm> ncclComm;
  std::shared_ptr<ExtNCCLComm> extNcclComm;

  // Create the unique NCCL ID and broadcast it
  ncclUniqueId ncclID;

  // reset log prefix to include group_desc
  logPrefix_ = createExtLogPrefix();

  #ifdef NCCL_COMM_DESCRIPTION
  // Pass process group name and description to NCCL communicator
  std::string commDesc = pg_desc_ + ':' + pg_uid_;
  options_->config.commDesc = strdup(commDesc.c_str());
  #endif

  // For batch_isend_irecv, ncclGroupStart() would be called upfront
  bool batchP2P = ncclActiveGroupCounter_ > 0;
  bool singleP2POp = isP2POp(opType, batchP2P);

  // Get the device index
  auto deviceIndex = device.index();
  at::cuda::OptionalCUDAGuard gpuGuard(device);

  // [Group Start/End Note] This is used to ensure that nccl communicator will
  // be created before communication primitives are called. Let's look at this
  // example: Using the batch_isend_irecv to send a tensor to a target process.
  // On the sender side, the corresponding underlying NCCL calls will look like
  //   ncclGroupStart() // This is in batch_isend_irecv
  //   ncclCommInitRank() // Inside NCCLComm::create
  //   ncclSend()
  //   ncclGroupEnd() // This is in batch_isend_irecv
  // With this pattern, the nccl communicator will be created in the last
  // ncclGroupEnd which means when ncclSend is processed, the passed
  // communicator argument is NULL which will lead to runtime error. So we need
  // to "close" all active nccl groups to ensure nccl communicator is actually
  // created before encountering any communication calls. This is why we need
  // the following for loop.
  for (const auto i : c10::irange(ncclActiveGroupCounter_)) {
    (void)i;
    // comms have not been initiated yet, so can only check in blocking-way
    C10D_NCCL_CHECK(ncclGroupEnd(), std::nullopt);
  }

  // GPU world size and GPU rank
  int numRanks = -1, rank = -1;

  if (!singleP2POp) {
    // Collective, all-to-all, or batch P2P
    numRanks = getSize();
    rank = getRank();
  } else if (isSendRecvSelf) {
    // Same process send and recv.
    numRanks = 1;
    rank = 0;
  } else {
    // For single point-to-point operation, there are only 2 processes
    // involved so the GPU rank is either 0 or 1.
    numRanks = 2;
    rank = p2pRank;
  }

  #ifdef NCCL_HAS_COMM_NONBLOCKING
  bool useNb = useNonblocking();
  options_->config.blocking = useNb ? 0 : 1;
  #endif

  #ifdef NCCL_HAS_COMM_SPLIT
  // // Use split to create a new communicator only if:
  // // 1. The parent comm is known; AND
  // // 2. The new comm is not for a point-to-point operation.
  // // ncclCommSplit() is a collective call, so it does not work for P2P
  // // operations.
  // if (options_->split_from && !singleP2POp) {
  //   // Find a valid, healthy communicator to split from if possible.
  //   std::lock_guard<std::mutex> lock(options_->split_from->mutex_);
  //   auto& other_comms = options_->split_from->devExtNCCLCommMap_;
  //   auto dit = other_comms.find(getKeyFromDevice(device));
  //   if (dit != other_comms.end()) {
  //     auto& parentComm = dit->second;
  //     if (parentComm != nullptr && !parentComm->isAborted()) {
  //       LOG(INFO) << logPrefix_ << "Splitting NCCL communicator from "
  //                 << parentComm->repr();
  //       ncclComm = NCCLComm::split(
  //           parentComm.get(),
  //           options_->split_color,
  //           rank,
  //           options_->config,
  //           options_->global_ranks_in_group);
  //     }
  //   }
  // }
  TORCH_CHECK(false, "ncclCommSplit is not supported in extended nccl backend")
  #endif

  // To simplify conditional nesting, just create the ncclComms[i]
  // entry if it hasn't been yet rather than untangling the
  // conditions that might have resulted in a split above.
  if (!extNcclComm) {
    if (getCvarBool(TORCH_NCCL_BCAST_UNIQUEID, true) && !isSendRecvSelf) {
      // For point-to-point communication, lower rank of the two will get unique
      // id.
      if (rank_ == 0 || (singleP2POp && p2pRank == 0)) {
        C10D_NCCL_CHECK(ncclGetUniqueId(&ncclID), std::nullopt);
      }

      // Broadcast so that each process can have a unique NCCL ID
      auto timeStarted = std::chrono::steady_clock::now();
      broadcastUniqueNCCLID(&ncclID, singleP2POp, deviceKey, p2pRank);
      auto timerDeltaMs =
          std::chrono::duration_cast<std::chrono::duration<double>>(
              std::chrono::steady_clock::now() - timeStarted)
              .count() *
          1000;
      LOG(INFO) << logPrefix_
                << "ProcessGroupNCCL broadcast unique ID through store took "
                << timerDeltaMs << " ms";
    }

  #ifdef NCCL_HAS_COMM_NONBLOCKING
    extNcclComm =
          ExtNCCLComm::create(numRanks, rank, ncclID, deviceIndex, options_->config);
  #else
    extNcclComm = ExtNCCLComm::create(numRanks, rank, ncclID, deviceIndex);
  #endif
  }

  // Creates the NCCL streams
  bool force_high = getCvarBool(TORCH_NCCL_HIGH_PRIORITY, false);
  auto streamVal = at::cuda::getStreamFromPool(
      options_->is_high_priority_stream || force_high);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    inInitializationExtCommMap_.emplace(deviceKey, extNcclComm);
  }

  FlightRecorder::get()->record_pg_ranks(
      std::make_tuple(pg_uid_, pg_desc_), extGroupRanks());

  RECORD_PARAM_COMMS(
      std::make_tuple(0, false), // seq
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      rank, // rank
      "init", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      globalRankStart_, // globalRankStart
      globalRankStride_, // globalRankStride
      size_ // worldSize
  ); 

  VLOG(2) << logPrefix_ << "ExtProcessGroupNCCL created extNcclComm_ "
          << extNcclComm->repr()
          << " on CUDA device: " << static_cast<int>(deviceIndex);

  // At this point NCCL should have been initialized, hence we can accurately
  // get the env value even if NCCL sets it by reading from nccl.conf file
  LOG(INFO) << logPrefix_
            << "NCCL_DEBUG: " << getCvarString({"NCCL_DEBUG"}, "N/A");

  // See [Group Start/End Note]
  for (const auto i : c10::irange(ncclActiveGroupCounter_)) {
    (void)i;
    C10D_NCCL_CHECK(ncclGroupStart(), std::nullopt);
  }

  ncclStreams_.emplace(deviceKey, streamVal);

  // Note: these events are created with the (default) cudaEventDisableTiming
  // flag This flag provides the best performance when used with
  // cudaStreamWaitEvent() and cudaEventQuery(). Since we here don't measure the
  // performance using cudaEvent, this should be set.
  // TODO(kwen2501): is ncclEvents_ used anywhere else?
  ncclEvents_.emplace(deviceKey, at::cuda::CUDAEvent(cudaEventDisableTiming));

  // Move the NCCL resource to cache
  auto it = inInitializationExtCommMap_.find(deviceKey);
  // A previous thread could've already removed devicesKey from
  // inInitializationExtCommMap_ and added it to devExtNCCLCommMap_
  if (it != inInitializationExtCommMap_.end()) {
    devExtNCCLCommMap_.emplace(deviceKey, std::move(it->second));
    inInitializationExtCommMap_.erase(deviceKey);

    // Now ncclComms are fully initialized.
    // Register all active CUDA memory segments in cache allocator to
    // the new NCCL communicators
    if (useTensorRegisterAllocatorHook_) {
      auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
      // Register the segment to a new NCCL communicator if on the same device
      for (const auto& segmentInfo : snapshot.segments) {
        TORCH_INTERNAL_ASSERT(
            segmentInfo.device == device.index(),
            "Mismatch between CUDA memory segment device and current device");
        extNcclComm->registerSegment(
            reinterpret_cast<void*>(segmentInfo.address),
            segmentInfo.total_size);
      }
    }
    // Record the mapping between extNcclComm and device index so that later
    // register hook can register a newly allocated segment to communicators on the same device.
    // NOTE: we need remove the communicator from this map when it is
    // destroyed, otherwise may register onto an invalid communicator.
    extNcclCommDevIdxMapMutex.lock();
    extNcclCommDevIdxMap.emplace(extNcclComm, device.index());
    extNcclCommDevIdxMapMutex.unlock();
  }

  it = devExtNCCLCommMap_.find(deviceKey);
  TORCH_INTERNAL_ASSERT(
      it != devExtNCCLCommMap_.end(), "Communicators not populated in cache!");
  return it->second;
}


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
    bool nanCheck
) {
  // Environment setting by the user may add onto collective call's option
  avoidRecordStreams |= avoidRecordStreams_;
  nanCheck &= enableNanCheck_;

  auto device = inputs[0].device();
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

  const auto key = getDeviceKey();
  // std::shared_ptr<NCCLComm> ncclComm = getNCCLComm(key);
  std::shared_ptr<ExtNCCLComm> extNcclComm = getExtNCCLComm(key);
  if (extNcclComm == nullptr) {
    // ncclComm = initNCCLComm(key, device, opType);
    extNcclComm = initExtNCCLComm(key, device, opType);
  }

  if (coalescing_state_ & ExtCoalActive) {
    if ((coalescing_state_ & ExtCoalColl) == 0) {
      // First op in coalesced operations
      seqCollective_++;
    }
    coalescing_state_ |= ExtCoalColl;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), EXT_MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = extNcclComm;
    } else {
      TORCH_CHECK(coalescedComm_ == extNcclComm, EXT_MULTI_DEVICE_ERROR_MSG);
    }
  }

  // Used many times below, so we stash the unordered_map lookup
  auto ncclStream = getNCCLStream();

  // First let NCCL streams wait for input tensors allocation streams
  syncStream(device, ncclEvents_[key], ncclStream);

  bool enqueue =
      !coalescing_state_ && capture_status == c10::cuda::CaptureStatus::None;
  auto work = initExtWork(
      device, rank_, opType, false, profilingTitle, inputs, outputs, enqueue);

  // Store references to outputs to be used by ExtWorkNCCL::result and operator<<.
  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  if (avoidRecordStreams) {
    /** NOTE: If set, ProcessGroupNCCL doesn't use recordStream calls to ensure
     * caching allocator safety for tensors used on both user-facing and internal comm streams.
     * Instead, it stashes live references to those tensors until after user-facing streams are synced with comm streams,
     * by recording the inputs to stashed_for_allocator_safety_ as below.
    */
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

  ncclComm_t comm = extNcclComm->getNcclComm();
  if (!avoidRecordStreams) {
    /** NOTE: Both `inputs' and `outputs' are created on a worker stream and used in different ncclStreams.
     * Hence, both must record the ncclStream to prevent being freed before the collective finishes.
     * We only record `inputs' here, and leave recording `outputs' to `fn' 
     * for operations where `inputs' and `outputs' are not the same.
     */
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
      extNcclComm->getNcclCommFailureReason());
#else
  C10D_NCCL_CHECK_TIMEOUT(
      fn(inputs[0], outputs[0], comm, ncclStream),
      comm,
      extNcclComm->getNcclCommFailureReason());
#endif

  post(ncclStream, work);

  // End event should only be recorded after the ncclGroupEnd()
  if (!coalescing_state_) {
    work->ncclEndEvent_->record(ncclStream);
  }
  work->extNcclComm_ = extNcclComm;

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
  assignTimeoutToExtWork(work, options_);
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
    // workEnqueue(work);
    extWorkEnqueue(work);
  } else {
    at::cuda::CUDAGraph::dec_pending_event_queries();
  }

  return work;  // ExtProcessGroupNCCL::ExtWorkNCCL
}

std::exception_ptr ExtProcessGroupNCCL::checkForNCCLErrorsInternal(
  std::shared_ptr<ExtNCCLComm>& extNcclComm
) {
  // Prioritize commFailureReason over checkForNcclError() result if
  // commFailureReason is set.
  auto commFailureReason = extNcclComm->getNcclCommFailureReason();
  if (commFailureReason != std::nullopt) {
    return std::make_exception_ptr(C10_BUILD_ERROR(
        DistBackendError,
        c10::str(
            "NCCL communicator encountered error set by ProcessGroupNCCL: ",
            *commFailureReason)));
  }
  ncclResult_t ncclAsyncErr = extNcclComm->checkForNcclError();
  // When nonblocking mode is enabled by TORCH_NCCL_USE_COMM_NONBLOCKING,
  // ncclInProgress could be returned when there are pending NCCL calls.
  // In this case, no exception should be thrown
  #ifdef NCCL_HAS_COMM_NONBLOCKING
  // ncclInProgress is defined only if NCCL_HAS_COMM_NONBLOCKING is defined
  if (ncclAsyncErr != ncclSuccess && ncclAsyncErr != ncclInProgress) {
  #else
  if (ncclAsyncErr != ncclSuccess) {
  #endif
    return std::make_exception_ptr(C10_BUILD_ERROR(
        DistBackendError,
        "NCCL error: " + ncclGetErrorWithVersion(ncclAsyncErr) + "\n" +
            getNcclErrorDetailStr(ncclAsyncErr)));
  }

  return nullptr;
}


void ExtProcessGroupNCCL::extWorkEnqueue(
  const c10::intrusive_ptr<ExtProcessGroupNCCL::ExtWorkNCCL>& work
) {
  // in blockingWait_ mode, we don't need watchdog thread, so no need to enqueue
  // the work
  if (!terminateProcessGroup_.load() && !blockingWait_) {
    std::lock_guard<std::mutex> lock(workMetaListMutex_);
    // Avoid view tensors to be processed in cleanup thread.
    // View tensors' destruction invokes autograd_meta, which
    // needs to be destructed in user thread. Otherwise will
    // get deadlock. Here we enqueue work without outputs_.
    extWorkMetaList_.emplace_back(*work);
    // update the PG status related to the last enqueued work
    pgStatus_->lastEnqueuedSeq = work->seq_;
    pgStatus_->lastEnqueuedWorkName = opTypeToString(work->opType_);
    pgStatus_->lastEnqueuedNumelIn = work->numelIn_;
    pgStatus_->lastEnqueuedNumelOut = work->numelOut_;
    lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
  }
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
        this->getSize() // worldSize
    );

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
          torch::cuda::nccl::all2all_single_equal_split(
              input, output, this->getSize(), comm, stream);
          return ncclSuccess;
        },
        [](at::cuda::CUDAStream&,
          c10::intrusive_ptr<ExtProcessGroupNCCL::ExtWorkNCCL>& work) {},
        [](at::cuda::CUDAStream&,
            c10::intrusive_ptr<ExtProcessGroupNCCL::ExtWorkNCCL>& work) {},
        OpType::ALLTOALL_BASE,
        "nccl:all_to_all"
      );
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
        this->getSize() // worldSize
      ); 

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
        },
        [](at::cuda::CUDAStream&,
          c10::intrusive_ptr<ExtProcessGroupNCCL::ExtWorkNCCL>& work) {},
        [](at::cuda::CUDAStream&,
            c10::intrusive_ptr<ExtProcessGroupNCCL::ExtWorkNCCL>& work) {},
        OpType::ALLTOALL_BASE,
        "nccl:all_to_all"
      );
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

  return work; // ProcessGroupNCCL::WorkNCCL
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
