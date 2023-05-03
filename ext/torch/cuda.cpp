#include <torch/torch.h>

#include <rice/rice.hpp>

#include "utils.h"

void init_cuda(Rice::Module& m) {
  Rice::define_module_under(m, "CUDA")
    .define_singleton_function("available?", &torch::cuda::is_available)
    .define_singleton_function("device_count", &torch::cuda::device_count)
    .define_singleton_function("manual_seed", &torch::cuda::manual_seed)
    .define_singleton_function("manual_seed_all", &torch::cuda::manual_seed_all);
}
