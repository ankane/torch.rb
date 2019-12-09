#include <torch/torch.h>
#include <rice/Object.hpp>
#include "templates.hpp"

Object tensor_tuple(std::tuple<torch::Tensor, torch::Tensor> x) {
  Array a;
  a.push(to_ruby<torch::Tensor>(std::get<0>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<1>(x)));
  return Object(a);
}
