#include <ATen/Context.h>
#include <ATen/DeviceAccelerator.h>
#include <torch/torch.h>

#include <rice/rice.hpp>

#include "utils.h"

namespace {

inline bool accelerator_available(c10::DeviceType device_type) {
  return at::globalContext()
      .getAcceleratorHooksInterface(device_type)
      .isAvailable();
}

} // namespace

void init_accelerator(Rice::Module& m) {
  auto rb_mAccelerator = Rice::define_module_under(m, "Accelerator");

  rb_mAccelerator.define_singleton_function(
      "_current_device",
      []() -> VALUE {
        auto acc = at::getAccelerator(false);
        if (!acc.has_value()) {
          return Rice::Nil;
        }
        torch::Device device(acc.value());
        return Rice::detail::To_Ruby<torch::Device>().convert(device);
      });

  rb_mAccelerator.define_singleton_function(
      "_is_available",
      []() {
        auto acc = at::getAccelerator(false);
        if (!acc.has_value()) {
          return false;
        }
        return accelerator_available(acc.value());
      });

  rb_mAccelerator.define_singleton_function(
      "_device_count",
      []() {
        auto acc = at::getAccelerator(false);
        if (!acc.has_value()) {
          return 0;
        }
        return static_cast<int>(at::accelerator::deviceCount());
      });
}
