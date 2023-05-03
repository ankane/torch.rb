#include <torch/torch.h>

#include <rice/rice.hpp>

#include "utils.h"

void init_random(Rice::Module& m) {
  Rice::define_module_under(m, "Random")
    .define_singleton_function(
      "initial_seed",
      []() {
        return at::detail::getDefaultCPUGenerator().current_seed();
      })
    .define_singleton_function(
      "seed",
      []() {
        // TODO set for CUDA when available
        auto generator = at::detail::getDefaultCPUGenerator();
        return generator.seed();
      });
}
