#include <torch/torch.h>

#include <rice/rice.hpp>

#include "utils.h"

#if defined(WITH_CUDA)
extern "C" {
int cudaGetDeviceCount(int* count);
int cudaSetDevice(int device);
}
#endif

void init_cuda(Rice::Module& m) {
  Rice::define_module_under(m, "CUDA")
    .define_singleton_function("available?", &torch::cuda::is_available)
    .define_singleton_function("device_count", &torch::cuda::device_count)
    .define_singleton_function("manual_seed", &torch::cuda::manual_seed)
    .define_singleton_function("manual_seed_all", &torch::cuda::manual_seed_all)
    .define_singleton_function(
      "set_device",
      [](int device_id) {
#if defined(WITH_CUDA)
        int count = 0;
        auto status = cudaGetDeviceCount(&count);
        if (status != 0) {
          rb_raise(rb_eRuntimeError, "cudaGetDeviceCount failed with code %d", status);
        }
        if (device_id < 0 || device_id >= count) {
          rb_raise(
              rb_eArgError,
              "Invalid device_id %d for CUDA (available devices: %d)",
              device_id,
              count);
        }
        status = cudaSetDevice(device_id);
        if (status != 0) {
          rb_raise(rb_eRuntimeError, "cudaSetDevice(%d) failed with code %d", device_id, status);
        }
#else
        rb_raise(rb_eRuntimeError, "Torch::CUDA.set_device requires CUDA support");
#endif
        return Rice::Nil;
      });
}
