#include <torch/torch.h>

#include <rice/rice.hpp>

#include "utils.h"

void init_generator(Rice::Module& m, Rice::Class& rb_cGenerator) {
  // https://github.com/pytorch/pytorch/blob/master/torch/csrc/Generator.cpp
  rb_cGenerator
    .define_singleton_function(
      "new",
      []() {
        // TODO support more devices
        return torch::make_generator<torch::CPUGeneratorImpl>();
      })
    .define_method(
      "device",
      [](torch::Generator& self) {
        return self.device();
      })
    .define_method(
      "initial_seed",
      [](torch::Generator& self) {
        return self.current_seed();
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
      })
    .define_method(
      "state",
      [](torch::Generator& self) {
        return self.get_state();
      })
    .define_method(
      "state=",
      [](torch::Generator& self, const torch::Tensor& state) {
        self.set_state(state);
      });

  THPGeneratorClass = rb_cGenerator.value();
}
