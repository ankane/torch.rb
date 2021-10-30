#include <torch/torch.h>

#include <rice/rice.hpp>

#include "utils.h"

void init_backends(Rice::Module& m) {
  auto rb_mBackends = Rice::define_module_under(m, "Backends");

  Rice::define_module_under(rb_mBackends, "OpenMP")
    .add_handler<torch::Error>(handle_error)
    .define_singleton_function("available?", &torch::hasOpenMP);

  Rice::define_module_under(rb_mBackends, "MKL")
    .add_handler<torch::Error>(handle_error)
    .define_singleton_function("available?", &torch::hasMKL);
}
