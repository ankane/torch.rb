#include <algorithm>
#include <chrono>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <rice/rice.hpp>
#include <rice/stl.hpp>

#include "utils.h"

#ifdef USE_C10D
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
using ProcessGroupPtr = c10::intrusive_ptr<::c10d::ProcessGroup>;

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
                  bool wait_for_workers) {
        ::c10d::TCPStoreOptions opts;
        opts.port = static_cast<uint16_t>(port);
        opts.isServer = is_master;
        opts.numWorkers = world_size;
        opts.waitWorkers = wait_for_workers;
        opts.timeout = std::chrono::milliseconds(timeout_millis);
        auto store = c10::make_intrusive<::c10d::TCPStore>(host, opts);
        return Rice::Data_Object<StoreWrapper>(new StoreWrapper(store), rb_cStore, true);
      });

  rb_mDistributed.define_singleton_function(
      "_create_file_store",
      [rb_cStore](const std::string& path, int world_size) {
        auto store = c10::make_intrusive<::c10d::FileStore>(path, world_size);
        return Rice::Data_Object<StoreWrapper>(new StoreWrapper(store), rb_cStore, true);
      });

#if !defined(_WIN32)
  rb_mDistributed.define_singleton_function(
      "_create_hash_store",
      [rb_cStore]() {
        auto store = c10::make_intrusive<::c10d::HashStore>();
        return Rice::Data_Object<StoreWrapper>(new StoreWrapper(store), rb_cStore, true);
      });
#endif

  rb_mDistributed.define_singleton_function(
      "_init_process_group",
      [rb_cProcessGroup](const std::string& backend,
                         StoreWrapper& store_wrapper,
                         int rank,
                         int world_size,
                         int64_t timeout_millis) {
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

        default_process_group = pg;
        return Rice::Data_Object<ProcessGroupWrapper>(new ProcessGroupWrapper(pg), rb_cProcessGroup, true);
      });

  rb_mDistributed.define_singleton_function(
      "_destroy_process_group",
      []() {
        default_process_group.reset();
        return Rice::Nil;
      });

  rb_mDistributed.define_singleton_function(
      "_initialized?",
      []() {
        return static_cast<bool>(default_process_group);
      });

  rb_mDistributed.define_singleton_function(
      "_default_process_group",
      [rb_cProcessGroup]() {
        if (!default_process_group) {
          return Rice::Nil;
        }
        return Rice::Data_Object<ProcessGroupWrapper>(new ProcessGroupWrapper(default_process_group), rb_cProcessGroup, true);
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

  rb_mDistributed.const_set("DEFAULT_TIMEOUT", INT2NUM(::c10d::kProcessGroupDefaultTimeout.count() / 1000));
#else
  rb_mDistributed.define_singleton_function("available?", []() { return false; });
#endif
}
