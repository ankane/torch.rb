#include <string>

#include <torch/torch.h>

#include <rice/rice.hpp>
#include <rice/stl.hpp>

#include "utils.h"

void init_device(Rice::Module& m) {
  auto rb_cDevice = Rice::define_class_under<torch::Device>(m, "Device");
  rb_cDevice
    .define_constructor(Rice::Constructor<torch::Device, const std::string&>())
    .define_method(
      "_index",
      [](torch::Device& self) {
        return self.index();
      })
    .define_method(
      "index?",
      [](torch::Device& self) {
        return self.has_index();
      })
    .define_method(
      "type",
      [](torch::Device& self) {
        std::stringstream s;
        s << self.type();
        return s.str();
      })
    .define_method(
      "to_s",
      [](torch::Device& self) {
        return self.str();
      });

  THPDeviceClass = rb_cDevice.value();
}
