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

inline VALUE wrap(const torch::Tensor& x) {
  return Rice::detail::To_Ruby<torch::Tensor>().convert(x);
}

inline VALUE wrap(const torch::Scalar& x) {
  return Rice::detail::To_Ruby<torch::Scalar>().convert(x);
}

inline VALUE wrap(const torch::ScalarType& x) {
  return Rice::detail::To_Ruby<torch::ScalarType>().convert(x);
}

inline VALUE wrap(const torch::QScheme& x) {
  return Rice::detail::To_Ruby<torch::QScheme>().convert(x);
}

inline VALUE wrap(const std::tuple<torch::Tensor, torch::Tensor>& x) {
  return Rice::detail::protect(
    rb_ary_new3,
    2,
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(x))
  );
}

inline VALUE wrap(const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& x) {
  return Rice::detail::protect(
    rb_ary_new3,
    3,
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<2>(x))
  );
}

inline VALUE wrap(const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>& x) {
  return Rice::detail::protect(
    rb_ary_new3,
    4,
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<2>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<3>(x))
  );
}

inline VALUE wrap(const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>& x) {
  return Rice::detail::protect(
    rb_ary_new3,
    5,
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<2>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<3>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<4>(x))
  );
}

inline VALUE wrap(const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t>& x) {
  return Rice::detail::protect(
    rb_ary_new3,
    4,
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<2>(x)),
    Rice::detail::To_Ruby<int64_t>().convert(std::get<3>(x))
  );
}

inline VALUE wrap(const std::tuple<torch::Tensor, torch::Tensor, double, int64_t>& x) {
  return Rice::detail::protect(
    rb_ary_new3,
    4,
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<torch::Tensor>().convert(std::get<1>(x)),
    Rice::detail::To_Ruby<double>().convert(std::get<2>(x)),
    Rice::detail::To_Ruby<int64_t>().convert(std::get<3>(x))
  );
}

inline VALUE wrap(const torch::TensorList& x) {
  auto a = Rice::detail::protect(rb_ary_new2, x.size());
  for (const auto& t : x) {
    Rice::detail::protect(rb_ary_push, a, Rice::detail::To_Ruby<torch::Tensor>().convert(t));
  }
  return a;
}

inline VALUE wrap(const std::tuple<double, double>& x) {
  return Rice::detail::protect(
    rb_ary_new3,
    2,
    Rice::detail::To_Ruby<double>().convert(std::get<0>(x)),
    Rice::detail::To_Ruby<double>().convert(std::get<1>(x))
  );
}
