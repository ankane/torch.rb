#include <torch/torch.h>

#include <rice/Module.hpp>

#include "utils.h"

void init_random(Rice::Module& m) {
  Rice::define_module_under(m, "Random")
    .add_handler<torch::Error>(handle_error)
    .define_singleton_method(
      "initial_seed",
      *[]() {
        return at::detail::getDefaultCPUGenerator().current_seed();
      })
    .define_singleton_method(
      "seed",
      *[]() {
        // TODO set for CUDA when available
        auto generator = at::detail::getDefaultCPUGenerator();
        return generator.seed();
      });
}
