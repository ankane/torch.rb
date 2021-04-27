#pragma once

#include <torch/torch.h>
#include <rice/rice.hpp>

inline VALUE wrap(bool x) {
  return Rice::detail::To_Ruby<bool>().convert(x);
}

inline VALUE wrap(int64_t x) {
  return Rice::detail::To_Ruby<int64_t>().convert(x);
}

inline VALUE wrap(double x) {
  return Rice::detail::To_Ruby<double>().convert(x);
}

inline VALUE wrap(torch::Tensor x) {
  return Rice::detail::To_Ruby<torch::Tensor>().convert(x);
}

inline VALUE wrap(torch::Scalar x) {
  return Rice::detail::To_Ruby<torch::Scalar>().convert(x);
}

inline VALUE wrap(torch::ScalarType x) {
  return Rice::detail::To_Ruby<torch::ScalarType>().convert(x);
}

inline VALUE wrap(torch::QScheme x) {
  return Rice::detail::To_Ruby<torch::QScheme>().convert(x);
}

inline VALUE wrap(std::tuple<torch::Tensor, torch::Tensor> x) {
  return rb_ary_new3(
    2,
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(x))
  );
}

inline VALUE wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> x) {
  return rb_ary_new3(
    3,
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<2>(x))
  );
}

inline VALUE wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> x) {
  return rb_ary_new3(
    4,
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<2>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<3>(x))
  );
}

inline VALUE wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> x) {
  return rb_ary_new3(
    5,
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<2>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<3>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<4>(x))
  );
}

inline VALUE wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> x) {
  return rb_ary_new3(
    4,
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<2>(x)),
    Rice::detail::To_Ruby<int64_t>().convert(std::get<3>(x))
  );
}

inline VALUE wrap(std::tuple<torch::Tensor, torch::Tensor, double, int64_t> x) {
  return rb_ary_new3(
    4,
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(x)),
    Rice::detail::To_Ruby<double>().convert(std::get<2>(x)),
    Rice::detail::To_Ruby<int64_t>().convert(std::get<3>(x))
  );
}

inline VALUE wrap(torch::TensorList x) {
  auto a = rb_ary_new2(x.size());
  for (auto t : x) {
    rb_ary_push(a, Rice::detail::To_Ruby<torch::Tensor>().convert(t));
  }
  return a;
}

inline VALUE wrap(std::tuple<double, double> x) {
  return rb_ary_new3(
    2,
    Rice::detail::To_Ruby<double>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<double>().convert(std::get<1>(x))
  );
}
