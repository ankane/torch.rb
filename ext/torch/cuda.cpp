#include <torch/torch.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAFunctions.h>
#endif

#include <rice/rice.hpp>

#include "utils.h"

void init_cuda(Rice::Module& m) {
  Rice::define_module_under(m, "CUDA")
    .define_singleton_function("available?", &torch::cuda::is_available)
    .define_singleton_function("device_count", &torch::cuda::device_count)
    .define_singleton_function("current_device", []() -> int {
#ifdef USE_CUDA
      if (torch::cuda::is_available()) {
        return static_cast<int>(c10::cuda::current_device());
      }
#endif
      return 0;
    })
    .define_singleton_function("set_device", [](int device) {
#ifdef USE_CUDA
      if (torch::cuda::is_available()) {
        c10::cuda::set_device(static_cast<c10::DeviceIndex>(device));
        return;
      }
#endif
      if (device != 0) {
        throw std::runtime_error("CUDA is not available");
      }
    })
    .define_singleton_function("synchronize", []() {
#ifdef USE_CUDA
      if (torch::cuda::is_available()) {
        c10::cuda::device_synchronize();
      }
#endif
    })
    .define_singleton_function("manual_seed", &torch::cuda::manual_seed)
    .define_singleton_function("manual_seed_all", &torch::cuda::manual_seed_all)
    .define_singleton_function("nccl_available?", []() -> bool {
#ifdef USE_NCCL
      return true;
#else
      return false;
#endif
    });
}
