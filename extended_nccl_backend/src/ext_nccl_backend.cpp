#include "../include/ext_nccl_backend.hpp"

// #include <torch/csrc/Exceptions.h>
// #include <torch/csrc/distributed/c10d/python_comm_hook.h>
// #include <torch/csrc/jit/python/pybind_utils.h>
// #include <torch/csrc/utils/object_ptr.h>
// #include <torch/csrc/utils/pybind.h>

// #include <torch/custom_class.h>

#ifndef USE_C10D_NCCL
#define USE_C10D_NCCL
#endif


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

// bool WorkExtNCCL::isCompleted() {
//   return true;
// }

// bool WorkExtNCCL::isSuccess() const {
//   return true;
// }

// bool WorkExtNCCL::wait(std::chrono::milliseconds /* unused */) {
//   return true;
// }

// c10::intrusive_ptr<c10::ivalue::Future> WorkExtNCCL::getFuture() {
//   return future_;
// }

// WorkExtNCCL::WorkExtNCCL(
//   std::string pgUID,
//   std::string pgDesc,
//   at::Device& device,
//   int rank,
//   OpType opType,
//   uint64_t seq,
//   bool isP2P,
//   const char* profilingTitle,
//   const std::optional<std::vector<at::Tensor>>& inputs,
//   bool enableTiming,
//   bool cudaEventCacheEnabled,
//   DebugLevel distDebugLevel
// ): ProcessGroupNCCL::WorkNCCL(pgUID, pgDesc, device, rank, opType, seq, isP2P, profilingTitle, inputs, enableTiming, cudaEventCacheEnabled, distDebugLevel) {}


// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
// ExtProcessGroupNCCL::ExtProcessGroupNCCL(int rank, int size)
//     : Backend(rank, size) {}

ExtProcessGroupNCCL::ExtProcessGroupNCCL(
  c10::intrusive_ptr<c10d::Store> store,
  int rank,
  int size) : ProcessGroupNCCL(store, rank, size) {}

ExtProcessGroupNCCL::~ExtProcessGroupNCCL() = default;

// // This is a dummy allgather that sets all output tensors to zero
// // Modify the implementation to conduct real communication asynchronously
// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::allgather(
//     std::vector<std::vector<at::Tensor>>& outputTensors,
//     std::vector<at::Tensor>& inputTensors,
//     const AllgatherOptions& /* unused */) {
//   for (auto& outputTensorVec : outputTensors) {
//       for (auto& outputTensor : outputTensorVec) {
//           outputTensor.zero_();
//       }
//   }

//   auto future = c10::make_intrusive<c10::ivalue::Future>(
//     c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
//   future->markCompleted(c10::IValue(outputTensors));
//   return c10::make_intrusive<WorkExtNCCL>(OpType::ALLGATHER, std::move(future));
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::_allgather_base(
//     at::Tensor& /* unused */,
//     at::Tensor& /* unused */,
//     const AllgatherOptions& /* unused */) {
//   throw std::runtime_error("not supported");
// }

// // This is a dummy allreduce that sets all output tensors to zero
// // Modify the implementation to conduct real communication asynchronously
// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::allreduce(
//     std::vector<at::Tensor>& tensors,
//     const AllreduceOptions& opts) {
//   for (auto& tensor : tensors) {
//       tensor.zero_();
//   }

//   auto future = c10::make_intrusive<c10::ivalue::Future>(
//     c10::ListType::create(c10::TensorType::get()));
//   future->markCompleted(c10::IValue(tensors));
//   return c10::make_intrusive<WorkExtNCCL>(OpType::ALLREDUCE, std::move(future));
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::allreduce_coalesced(
//     std::vector<at::Tensor>& /* unused */,
//     const AllreduceCoalescedOptions& /* unused */) {
//   throw std::runtime_error("not supported");
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::alltoall(
//     std::vector<at::Tensor>& /* unused */,
//     std::vector<at::Tensor>& /* unused */,
//     const AllToAllOptions& /* unused */) {
//   throw std::runtime_error("not supported");
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::alltoall_base(
//     at::Tensor& outputTensor,
//     at::Tensor& inputTensor,
//     std::vector<int64_t>& outputSplitSizes,
//     std::vector<int64_t>& inputSplitSizes,
//     const AllToAllOptions& /* unused */) {
//   throw std::runtime_error("not supported");
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::barrier(
//     const BarrierOptions& /* unused */) {
//   throw std::runtime_error("not supported");
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::broadcast(
//     std::vector<at::Tensor>& tensors,
//     const BroadcastOptions& opts) {
//   throw std::runtime_error("not supported");
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::gather(
//     std::vector<std::vector<at::Tensor>>& /* unused */,
//     std::vector<at::Tensor>& /* unused */,
//     const GatherOptions& /* unused */) {
//   throw std::runtime_error("not supported");
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::reduce(
//     std::vector<at::Tensor>& /* unused */,
//     const ReduceOptions& /* unused */) {
//   throw std::runtime_error("not supported");
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::reduce_scatter(
//     std::vector<at::Tensor>& /* unused */,
//     std::vector<std::vector<at::Tensor>>& /* unused */,
//     const ReduceScatterOptions& /* unused */) {
//   throw std::runtime_error("not supported");
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::scatter(
//     std::vector<at::Tensor>& /* unused */,
//     std::vector<std::vector<at::Tensor>>& /* unused */,
//     const ScatterOptions& /* unused */) {
//   throw std::runtime_error("not supported");
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::send(
//     std::vector<at::Tensor>& tensors,
//     int dstRank,
//     int tag) {
//   throw std::runtime_error("not supported");
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::recv(
//     std::vector<at::Tensor>& tensors,
//     int srcRank,
//     int tag) {
//   throw std::runtime_error("not supported");
// }

// c10::intrusive_ptr<Work> ExtProcessGroupNCCL::recvAnysource(
//     std::vector<at::Tensor>& tensors,
//     int tag) {
//   throw std::runtime_error("not supported");
// }

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
            // gil_scoped_release is not safe as a call_guard in init.
            // https://github.com/pybind/pybind11/issues/5473
            py::gil_scoped_release nogil;
            return c10::make_intrusive<ExtProcessGroupNCCL>(
                store, rank, size);
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          "Create ExtProcessGroupNCCL instance");
}

} // namespace c10d
