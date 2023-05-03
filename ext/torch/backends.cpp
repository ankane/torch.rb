#include <torch/torch.h>

#include <rice/rice.hpp>

#include "utils.h"

void init_backends(Rice::Module& m) {
  auto rb_mBackends = Rice::define_module_under(m, "Backends");

  Rice::define_module_under(rb_mBackends, "OpenMP")
    .define_singleton_function("available?", &torch::hasOpenMP);

  Rice::define_module_under(rb_mBackends, "MKL")
    .define_singleton_function("available?", &torch::hasMKL);

  Rice::define_module_under(rb_mBackends, "MPS")
    .define_singleton_function("available?", &torch::hasMPS);
}
