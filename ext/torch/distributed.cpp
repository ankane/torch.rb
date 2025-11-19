#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>
#if defined(USE_C10D) && defined(USE_C10D_NCCL)
#include <torch/cuda.h>
#include <c10/cuda/CUDAFunctions.h>
#endif

#include <rice/rice.hpp>
#include <rice/stl.hpp>

#include "utils.h"

#ifdef USE_C10D
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/FileStore.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#endif

#if defined(USE_C10D) && defined(USE_C10D_NCCL)
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif

#if defined(USE_C10D) && !defined(_WIN32)
#include <torch/csrc/distributed/c10d/HashStore.hpp>
#endif

namespace {

#ifdef USE_C10D

using StorePtr = c10::intrusive_ptr<::c10d::Store>;
using ProcessGroupPtr = c10::intrusive_ptr<::c10d::Backend>;

struct StoreWrapper {
  StoreWrapper() = default;
  explicit StoreWrapper(StorePtr store) : store_(std::move(store)) {}

  StorePtr store_;
};

struct ProcessGroupWrapper {
  ProcessGroupWrapper() = default;
  explicit ProcessGroupWrapper(ProcessGroupPtr pg) : pg_(std::move(pg)) {}

  ProcessGroupPtr pg_;
};

ProcessGroupPtr default_process_group;
std::once_flag default_pg_cleanup_once;

void shutdown_default_process_group() {
  if (default_process_group) {
    try {
      default_process_group->shutdown();
    } catch (...) {
      // best effort; ensure reset still happens
    }
    default_process_group.reset();
  }
}

void register_default_pg_cleanup() {
  std::call_once(default_pg_cleanup_once, []() {
    std::atexit([]() { shutdown_default_process_group(); });
  });
}

ProcessGroupPtr resolve_process_group(Rice::Object pg_obj) {
  if (pg_obj.is_nil()) {
    if (!default_process_group) {
      rb_raise(rb_eRuntimeError, "Distributed process group not initialized");
    }
    return default_process_group;
  }
  auto& wrapper = Rice::detail::From_Ruby<ProcessGroupWrapper&>().convert(pg_obj.value());
  if (!wrapper.pg_) {
    rb_raise(rb_eRuntimeError, "Invalid process group");
  }
  return wrapper.pg_;
}

int reduce_op_from_int(int code) {
  if (code < 0 || code > static_cast<int>(::c10d::ReduceOp::UNUSED)) {
    rb_raise(rb_eArgError, "Unknown reduce op code");
  }
  return code;
}

#endif

} // namespace

void init_distributed(Rice::Module& m) {
  auto rb_mDistributed = Rice::define_module_under(m, "Distributed");
#ifdef USE_C10D
  register_default_pg_cleanup();
  rb_mDistributed.define_singleton_function("available?", []() { return true; });

  auto rb_cStore = Rice::define_class_under<StoreWrapper>(rb_mDistributed, "Store");
  rb_cStore.define_method(
      "_native?",
      [](StoreWrapper& self) {
        return static_cast<bool>(self.store_);
      });

  auto rb_cProcessGroup = Rice::define_class_under<ProcessGroupWrapper>(rb_mDistributed, "ProcessGroup")
    .define_method(
      "rank",
      [](ProcessGroupWrapper& self) {
        return self.pg_ ? self.pg_->getRank() : -1;
      })
    .define_method(
      "size",
      [](ProcessGroupWrapper& self) {
        return self.pg_ ? self.pg_->getSize() : 0;
      })
    .define_method(
      "backend",
      [](ProcessGroupWrapper& self) {
        if (!self.pg_) {
          return std::string();
        }
        return self.pg_->getBackendName();
      });

  rb_mDistributed.define_singleton_function(
      "_create_tcp_store",
      [rb_cStore](const std::string& host,
                  int port,
                  int world_size,
                  bool is_master,
                  int64_t timeout_millis,
                  bool wait_for_workers) -> Rice::Object {
        ::c10d::TCPStoreOptions opts;
        opts.port = static_cast<uint16_t>(port);
        opts.isServer = is_master;
        opts.numWorkers = world_size;
        opts.waitWorkers = wait_for_workers;
        opts.timeout = std::chrono::milliseconds(timeout_millis);
        auto store = c10::make_intrusive<::c10d::TCPStore>(host, opts);
        // Pass ownership first, then the Ruby class so Rice doesn't treat the class as the owner flag
        return Rice::Data_Object<StoreWrapper>(new StoreWrapper(store), true, rb_cStore);
      });

  rb_mDistributed.define_singleton_function(
      "_create_file_store",
      [rb_cStore](const std::string& path, int world_size) -> Rice::Object {
        auto store = c10::make_intrusive<::c10d::FileStore>(path, world_size);
        return Rice::Data_Object<StoreWrapper>(new StoreWrapper(store), true, rb_cStore);
      });

#if !defined(_WIN32)
  rb_mDistributed.define_singleton_function(
      "_create_hash_store",
      [rb_cStore]() -> Rice::Object {
        auto store = c10::make_intrusive<::c10d::HashStore>();
        return Rice::Data_Object<StoreWrapper>(new StoreWrapper(store), true, rb_cStore);
      });
#endif

  rb_mDistributed.define_singleton_function(
      "_init_process_group",
      [rb_cProcessGroup](const std::string& backend,
                         StoreWrapper& store_wrapper,
                         int rank,
                         int world_size,
                         int64_t timeout_millis,
                         int device_id) -> Rice::Object {
        StorePtr store = store_wrapper.store_;
        if (!store) {
          rb_raise(rb_eArgError, "Store is required for init_process_group");
        }

        std::string backend_lower = backend;
        std::transform(backend_lower.begin(), backend_lower.end(), backend_lower.begin(), ::tolower);

        ProcessGroupPtr pg;
        if (backend_lower == "gloo") {
#ifdef USE_C10D_GLOO
          auto options = ::c10d::ProcessGroupGloo::Options::create();
          options->timeout = std::chrono::milliseconds(timeout_millis);
          options->devices.push_back(::c10d::ProcessGroupGloo::createDefaultDevice());
          pg = c10::make_intrusive<::c10d::ProcessGroupGloo>(store, rank, world_size, options);
#else
          rb_raise(rb_eRuntimeError, "Gloo backend is not available in this build");
#endif
        } else if (backend_lower == "nccl") {
#if defined(USE_C10D_NCCL)
          auto options = c10::make_intrusive<::c10d::ProcessGroupNCCL::Options>();
          pg = c10::make_intrusive<::c10d::ProcessGroupNCCL>(store, rank, world_size, options);
#else
          rb_raise(rb_eRuntimeError, "NCCL backend is not available in this build");
#endif
        } else {
          rb_raise(rb_eArgError, "Unsupported backend: %s", backend.c_str());
        }

        if (device_id >= 0 && backend_lower == "nccl") {
#if defined(USE_C10D_NCCL)
          if (!torch::cuda::is_available()) {
            rb_raise(rb_eRuntimeError, "CUDA is not available for NCCL backend");
          }
          auto device_count = torch::cuda::device_count();
          if (device_id >= static_cast<int>(device_count)) {
            rb_raise(
                rb_eArgError,
                "Invalid device_id %d for NCCL backend (available devices: %d)",
                device_id,
                static_cast<int>(device_count));
          }
          c10::cuda::set_device(device_id);
          pg->setBoundDeviceId(c10::Device(c10::kCUDA, device_id));
#endif
        }

        default_process_group = pg;
        return Rice::Data_Object<ProcessGroupWrapper>(new ProcessGroupWrapper(pg), true, rb_cProcessGroup);
      });

  rb_mDistributed.define_singleton_function(
      "_destroy_process_group",
      []() {
        shutdown_default_process_group();
        return Rice::Nil;
      });

  rb_mDistributed.define_singleton_function(
      "_initialized?",
      []() {
        return static_cast<bool>(default_process_group);
      });

  rb_mDistributed.define_singleton_function(
      "_default_process_group",
      [rb_cProcessGroup]() -> Rice::Object {
        if (!default_process_group) {
          return Rice::Nil;
        }
        return Rice::Data_Object<ProcessGroupWrapper>(new ProcessGroupWrapper(default_process_group), true, rb_cProcessGroup);
      });

  rb_mDistributed.define_singleton_function(
      "_get_world_size",
      [](Rice::Object pg_obj) {
        auto pg = resolve_process_group(pg_obj);
        return pg->getSize();
      });

  rb_mDistributed.define_singleton_function(
      "_get_rank",
      [](Rice::Object pg_obj) {
        auto pg = resolve_process_group(pg_obj);
        return pg->getRank();
      });

  rb_mDistributed.define_singleton_function(
      "_barrier",
      [](Rice::Object pg_obj) {
        auto pg = resolve_process_group(pg_obj);
        ::c10d::BarrierOptions opts;
        auto work = pg->barrier(opts);
        work->wait();
        return Rice::Nil;
      });

  rb_mDistributed.define_singleton_function(
      "_all_reduce",
      [](torch::Tensor& tensor, int op_code, Rice::Object pg_obj) {
        auto pg = resolve_process_group(pg_obj);
        ::c10d::AllreduceOptions opts;
        opts.reduceOp = ::c10d::ReduceOp(static_cast<::c10d::ReduceOp::RedOpType>(reduce_op_from_int(op_code)));
        std::vector<at::Tensor> tensors{tensor};
        auto work = pg->allreduce(tensors, opts);
        work->wait();
        return tensor;
      });

  rb_mDistributed.define_singleton_function(
      "_broadcast",
      [](torch::Tensor& tensor, int src, Rice::Object pg_obj) {
        auto pg = resolve_process_group(pg_obj);
        ::c10d::BroadcastOptions opts;
        opts.rootRank = src;
        std::vector<at::Tensor> tensors{tensor};
        auto work = pg->broadcast(tensors, opts);
        work->wait();
        return tensor;
      });

  rb_mDistributed.define_singleton_function(
      "_register_ddp_hook",
      [](torch::Tensor& tensor, ProcessGroupWrapper& pg_wrapper, int world_size) -> unsigned {
        if (!pg_wrapper.pg_) {
          rb_raise(rb_eArgError, "Process group is required for DDP hook registration");
        }
        if (world_size <= 0) {
          rb_raise(rb_eArgError, "world_size must be positive");
        }

        auto pg = pg_wrapper.pg_;
        // Register a native autograd hook that all-reduces gradients and scales
        // them by the world size. This avoids calling back into Ruby from
        // autograd worker threads.
        unsigned handle = tensor.register_hook([pg, world_size](const at::Tensor& grad) {
          ::c10d::AllreduceOptions opts;
          opts.reduceOp = ::c10d::ReduceOp::SUM;
          std::vector<at::Tensor> tensors{grad};
          auto work = pg->allreduce(tensors, opts);
          work->wait();
          grad.div_(static_cast<double>(world_size));
          return grad;
        });

        return handle;
      });

  auto rb_mReduceOp = Rice::define_module_under(rb_mDistributed, "ReduceOp");
  rb_mReduceOp.const_set("SUM", INT2NUM(static_cast<int>(::c10d::ReduceOp::SUM)));
  rb_mReduceOp.const_set("AVG", INT2NUM(static_cast<int>(::c10d::ReduceOp::AVG)));
  rb_mReduceOp.const_set("PRODUCT", INT2NUM(static_cast<int>(::c10d::ReduceOp::PRODUCT)));
  rb_mReduceOp.const_set("MIN", INT2NUM(static_cast<int>(::c10d::ReduceOp::MIN)));
  rb_mReduceOp.const_set("MAX", INT2NUM(static_cast<int>(::c10d::ReduceOp::MAX)));
  rb_mReduceOp.const_set("BAND", INT2NUM(static_cast<int>(::c10d::ReduceOp::BAND)));
  rb_mReduceOp.const_set("BOR", INT2NUM(static_cast<int>(::c10d::ReduceOp::BOR)));
  rb_mReduceOp.const_set("BXOR", INT2NUM(static_cast<int>(::c10d::ReduceOp::BXOR)));
  rb_mReduceOp.const_set("PREMUL_SUM", INT2NUM(static_cast<int>(::c10d::ReduceOp::PREMUL_SUM)));

  rb_mDistributed.const_set("DEFAULT_TIMEOUT", INT2NUM(::kProcessGroupDefaultTimeout.count() / 1000));
#else
  rb_mDistributed.define_singleton_function("available?", []() { return false; });
#endif
}
