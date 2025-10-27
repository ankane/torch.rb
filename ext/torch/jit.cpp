#include <torch/torch.h>
#include <torch/script.h>

#include <rice/rice.hpp>

#include "utils.h"

void init_jit(Rice::Module& m, Rice::Class& rb_cScriptModule) {
  auto rb_mJit = Rice::define_module_under(m, "Jit");

  rb_mJit
    .define_singleton_function(
      "load",
      [](const std::string& path, Rice::Object device_obj) {
        c10::optional<torch::Device> device = c10::nullopt;
        if (!device_obj.is_nil()) {
          device = Rice::detail::From_Ruby<torch::Device>().convert(device_obj.value());
        }
        return torch::jit::load(path, device);
      },
      Rice::Arg("path"), Rice::Arg("device") = Rice::Object());

  rb_cScriptModule
    .define_method(
      "forward",
      [](torch::jit::script::Module& self, Rice::Array args) {
        std::vector<torch::jit::IValue> inputs;
        for (auto arg : args) {
          inputs.push_back(Rice::detail::From_Ruby<torch::Tensor>().convert(arg.value()));
        }

        auto output = self.forward(inputs);

        if (output.isTensor()) {
          return Rice::Object(Rice::detail::To_Ruby<torch::Tensor>().convert(output.toTensor()));
        } else if (output.isTuple()) {
          auto tuple = output.toTuple();
          Rice::Array result;
          for (const auto& elem : tuple->elements()) {
            if (elem.isTensor()) {
              result.push(Rice::Object(Rice::detail::To_Ruby<torch::Tensor>().convert(elem.toTensor())), false);
            } else {
              result.push(Rice::Object(Rice::detail::To_Ruby<torch::IValue>().convert(elem)), false);
            }
          }
          return Rice::Object(result);
        } else {
          return Rice::Object(Rice::detail::To_Ruby<torch::IValue>().convert(output));
        }
      })
    .define_method(
      "eval",
      [](torch::jit::script::Module& self) {
        self.eval();
        return self;
      })
    .define_method(
      "to",
      [](torch::jit::script::Module& self, Rice::Object device_obj) {
        c10::optional<torch::Device> device = c10::nullopt;

        if (!device_obj.is_nil()) {
          device = Rice::detail::From_Ruby<torch::Device>().convert(device_obj.value());
        }

        if (device.has_value()) {
          self.to(device.value());
        }

        return self;
      },
      Rice::Arg("device") = Rice::Object())
    .define_method(
      "save",
      [](torch::jit::script::Module& self, const std::string& path) {
        self.save(path);
      })
    .define_method(
      "parameters",
      [](torch::jit::script::Module& self, bool recurse) {
        Rice::Array params;
        for (const auto& param : self.parameters(recurse)) {
          params.push(Rice::Object(Rice::detail::To_Ruby<torch::Tensor>().convert(param)), false);
        }
        return params;
      },
      Rice::Arg("recurse") = true);
}
