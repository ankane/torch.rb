#include <torch/torch.h>

#include <rice/Module.hpp>

#include "utils.h"

void init_cuda(Rice::Module& m) {
  Rice::define_module_under(m, "CUDA")
    .add_handler<torch::Error>(handle_error)
    .define_singleton_method("available?", &torch::cuda::is_available)
    .define_singleton_method("device_count", &torch::cuda::device_count)
    .define_singleton_method("manual_seed", &torch::cuda::manual_seed)
    .define_singleton_method("manual_seed_all", &torch::cuda::manual_seed_all);
}
