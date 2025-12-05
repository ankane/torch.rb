#include <torch/torch.h>
#ifdef HAVE_C10_CUDA
#include <c10/cuda/CUDAFunctions.h>
#endif

#include <rice/rice.hpp>

#include "utils.h"

void init_cuda(Rice::Module& m) {
  auto rb_mCUDA = Rice::define_module_under(m, "CUDA");

  rb_mCUDA
    .define_singleton_function("available?", &torch::cuda::is_available)
    .define_singleton_function("device_count", &torch::cuda::device_count)
    .define_singleton_function("manual_seed", &torch::cuda::manual_seed)
    .define_singleton_function("manual_seed_all", &torch::cuda::manual_seed_all);

#ifdef HAVE_C10_CUDA
  rb_mCUDA.define_singleton_function(
      "set_device",
      [](int device_id) {
        c10::cuda::set_device(device_id);
        return Rice::Nil;
      });
#else
  rb_mCUDA.define_singleton_function(
      "set_device",
      [](int) {
        rb_raise(
            rb_eRuntimeError,
            "c10 CUDA support is not available in this build; set_device cannot be used");
        return Rice::Nil;
      });
#endif
}
