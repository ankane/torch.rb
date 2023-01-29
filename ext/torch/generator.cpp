#include <torch/torch.h>

#include <rice/rice.hpp>

#include "utils.h"

void init_generator(Rice::Module& m, Rice::Class& rb_cGenerator) {
  // https://github.com/pytorch/pytorch/blob/master/torch/csrc/Generator.cpp
  rb_cGenerator
    .add_handler<torch::Error>(handle_error)
    .define_singleton_function(
      "new",
      []() {
        // TODO support more devices
        return torch::make_generator<torch::CPUGeneratorImpl>();
      })
    .define_method(
      "manual_seed",
      [](torch::Generator& self, uint64_t seed) {
        self.set_current_seed(seed);
        return self;
      })
    .define_method(
      "seed",
      [](torch::Generator& self) {
        return self.seed();
      });

  THPGeneratorClass = rb_cGenerator.value();
}
