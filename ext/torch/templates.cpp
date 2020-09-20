#include <torch/torch.h>
#include <rice/Object.hpp>
#include "templates.hpp"

Object wrap(bool x) {
  return to_ruby<bool>(x);
}

Object wrap(int64_t x) {
  return to_ruby<int64_t>(x);
}

Object wrap(double x) {
  return to_ruby<double>(x);
}

Object wrap(torch::Tensor x) {
  return to_ruby<torch::Tensor>(x);
}

Object wrap(torch::Scalar x) {
  return to_ruby<torch::Scalar>(x);
}

Object wrap(torch::ScalarType x) {
  return to_ruby<torch::ScalarType>(x);
}

Object wrap(torch::QScheme x) {
  return to_ruby<torch::QScheme>(x);
}

Object wrap(std::tuple<torch::Tensor, torch::Tensor> x) {
  Array a;
  a.push(to_ruby<torch::Tensor>(std::get<0>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<1>(x)));
  return Object(a);
}

Object wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> x) {
  Array a;
  a.push(to_ruby<torch::Tensor>(std::get<0>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<1>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<2>(x)));
  return Object(a);
}

Object wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> x) {
  Array a;
  a.push(to_ruby<torch::Tensor>(std::get<0>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<1>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<2>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<3>(x)));
  return Object(a);
}

Object wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> x) {
  Array a;
  a.push(to_ruby<torch::Tensor>(std::get<0>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<1>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<2>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<3>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<4>(x)));
  return Object(a);
}

Object wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> x) {
  Array a;
  a.push(to_ruby<torch::Tensor>(std::get<0>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<1>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<2>(x)));
  a.push(to_ruby<int64_t>(std::get<3>(x)));
  return Object(a);
}

Object wrap(std::tuple<torch::Tensor, torch::Tensor, double, int64_t> x) {
  Array a;
  a.push(to_ruby<torch::Tensor>(std::get<0>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<1>(x)));
  a.push(to_ruby<double>(std::get<2>(x)));
  a.push(to_ruby<int64_t>(std::get<3>(x)));
  return Object(a);
}

Object wrap(std::vector<torch::Tensor> x) {
  Array a;
  for (auto& t : x) {
    a.push(to_ruby<torch::Tensor>(t));
  }
  return Object(a);
}
