#include <torch/torch.h>

#include <rice/rice.hpp>

#include "nn_functions.h"
#include "templates.h"
#include "utils.h"

// need to make a distinction between parameters and tensors
class Parameter: public torch::autograd::Variable {
  public:
    Parameter(Tensor&& t) : torch::autograd::Variable(t) { }
};

void init_nn(Rice::Module& m) {
  auto rb_mNN = Rice::define_module_under(m, "NN");
  rb_mNN.add_handler<torch::Error>(handle_error);
  add_nn_functions(rb_mNN);

  Rice::define_module_under(rb_mNN, "Init")
    .add_handler<torch::Error>(handle_error)
    .define_singleton_function(
      "_calculate_gain",
      [](NonlinearityType nonlinearity, double param) {
        return torch::nn::init::calculate_gain(nonlinearity, param);
      })
    .define_singleton_function(
      "_uniform!",
      [](Tensor tensor, double low, double high) {
        return torch::nn::init::uniform_(tensor, low, high);
      })
    .define_singleton_function(
      "_normal!",
      [](Tensor tensor, double mean, double std) {
        return torch::nn::init::normal_(tensor, mean, std);
      })
    .define_singleton_function(
      "_constant!",
      [](Tensor tensor, Scalar value) {
        return torch::nn::init::constant_(tensor, value);
      })
    .define_singleton_function(
      "_ones!",
      [](Tensor tensor) {
        return torch::nn::init::ones_(tensor);
      })
    .define_singleton_function(
      "_zeros!",
      [](Tensor tensor) {
        return torch::nn::init::zeros_(tensor);
      })
    .define_singleton_function(
      "_eye!",
      [](Tensor tensor) {
        return torch::nn::init::eye_(tensor);
      })
    .define_singleton_function(
      "_dirac!",
      [](Tensor tensor) {
        return torch::nn::init::dirac_(tensor);
      })
    .define_singleton_function(
      "_xavier_uniform!",
      [](Tensor tensor, double gain) {
        return torch::nn::init::xavier_uniform_(tensor, gain);
      })
    .define_singleton_function(
      "_xavier_normal!",
      [](Tensor tensor, double gain) {
        return torch::nn::init::xavier_normal_(tensor, gain);
      })
    .define_singleton_function(
      "_kaiming_uniform!",
      [](Tensor tensor, double a, FanModeType mode, NonlinearityType nonlinearity) {
        return torch::nn::init::kaiming_uniform_(tensor, a, mode, nonlinearity);
      })
    .define_singleton_function(
      "_kaiming_normal!",
      [](Tensor tensor, double a, FanModeType mode, NonlinearityType nonlinearity) {
        return torch::nn::init::kaiming_normal_(tensor, a, mode, nonlinearity);
      })
    .define_singleton_function(
      "_orthogonal!",
      [](Tensor tensor, double gain) {
        return torch::nn::init::orthogonal_(tensor, gain);
      })
    .define_singleton_function(
      "_sparse!",
      [](Tensor tensor, double sparsity, double std) {
        return torch::nn::init::sparse_(tensor, sparsity, std);
      });

  Rice::define_class_under<Parameter, torch::Tensor>(rb_mNN, "Parameter")
    .add_handler<torch::Error>(handle_error)
    .define_method(
      "grad",
      [](Parameter& self) {
        auto grad = self.grad();
        return grad.defined() ? Object(Rice::detail::To_Ruby<torch::Tensor>().convert(grad)) : Nil;
      })
    .define_method(
      "grad=",
      [](Parameter& self, torch::Tensor& grad) {
        self.mutable_grad() = grad;
      })
    .define_singleton_function(
      "_make_subclass",
      [](Tensor& rd, bool requires_grad) {
        auto data = rd.detach();
        data.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
        auto var = data.set_requires_grad(requires_grad);
        return Parameter(std::move(var));
      });
}
