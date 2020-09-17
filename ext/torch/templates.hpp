#pragma once

#ifdef isfinite
#undef isfinite
#endif

#include <rice/Array.hpp>
#include <rice/Object.hpp>

using namespace Rice;

using torch::Device;
using torch::Scalar;
using torch::ScalarType;
using torch::Tensor;

// need to wrap torch::IntArrayRef() since
// it doesn't own underlying data
class IntArrayRef {
  std::vector<int64_t> vec;
  public:
    IntArrayRef(Object o) {
      Array a = Array(o);
      for (size_t i = 0; i < a.size(); i++) {
        vec.push_back(from_ruby<int64_t>(a[i]));
      }
    }
    operator torch::IntArrayRef() {
      return torch::IntArrayRef(vec);
    }
};

template<>
inline
IntArrayRef from_ruby<IntArrayRef>(Object x)
{
  return IntArrayRef(x);
}

class TensorList {
  std::vector<torch::Tensor> vec;
  public:
    TensorList(Object o) {
      Array a = Array(o);
      for (size_t i = 0; i < a.size(); i++) {
        vec.push_back(from_ruby<torch::Tensor>(a[i]));
      }
    }
    operator torch::TensorList() {
      return torch::TensorList(vec);
    }
};

template<>
inline
TensorList from_ruby<TensorList>(Object x)
{
  return TensorList(x);
}

class FanModeType {
  std::string s;
  public:
    FanModeType(Object o) {
      s = String(o).str();
    }
    operator torch::nn::init::FanModeType() {
      if (s == "fan_in") {
        return torch::kFanIn;
      } else if (s == "fan_out") {
        return torch::kFanOut;
      } else {
        throw std::runtime_error("Unsupported nonlinearity type: " + s);
      }
    }
};

template<>
inline
FanModeType from_ruby<FanModeType>(Object x)
{
  return FanModeType(x);
}

class NonlinearityType {
  std::string s;
  public:
    NonlinearityType(Object o) {
      s = String(o).str();
    }
    operator torch::nn::init::NonlinearityType() {
      if (s == "linear") {
        return torch::kLinear;
      } else if (s == "conv1d") {
        return torch::kConv1D;
      } else if (s == "conv2d") {
        return torch::kConv2D;
      } else if (s == "conv3d") {
        return torch::kConv3D;
      } else if (s == "conv_transpose1d") {
        return torch::kConvTranspose1D;
      } else if (s == "conv_transpose2d") {
        return torch::kConvTranspose2D;
      } else if (s == "conv_transpose3d") {
        return torch::kConvTranspose3D;
      } else if (s == "sigmoid") {
        return torch::kSigmoid;
      } else if (s == "tanh") {
        return torch::kTanh;
      } else if (s == "relu") {
        return torch::kReLU;
      } else if (s == "leaky_relu") {
        return torch::kLeakyReLU;
      } else {
        throw std::runtime_error("Unsupported nonlinearity type: " + s);
      }
    }
};

template<>
inline
NonlinearityType from_ruby<NonlinearityType>(Object x)
{
  return NonlinearityType(x);
}

class MyReduction {
  Object value;
  public:
    MyReduction(Object o) {
      value = o;
    }
    operator int64_t() {
      if (value.is_nil()) {
        return torch::Reduction::None;
      }

      std::string s = String(value).str();
      if (s == "mean") {
        return torch::Reduction::Mean;
      } else if (s == "sum") {
        return torch::Reduction::Sum;
      } else if (s == "none") {
        return torch::Reduction::None;
      } else {
        throw std::runtime_error("Unsupported reduction: " + s);
      }
    }
};

template<>
inline
MyReduction from_ruby<MyReduction>(Object x)
{
  return MyReduction(x);
}

class OptionalTensor {
  Object value;
  public:
    OptionalTensor(Object o) {
      value = o;
    }
    operator torch::Tensor() {
      if (value.is_nil()) {
        return {};
      }
      return from_ruby<torch::Tensor>(value);
    }
};

template<>
inline
Scalar from_ruby<Scalar>(Object x)
{
  if (x.rb_type() == T_FIXNUM) {
    return torch::Scalar(from_ruby<int64_t>(x));
  } else {
    return torch::Scalar(from_ruby<double>(x));
  }
}

template<>
inline
OptionalTensor from_ruby<OptionalTensor>(Object x)
{
  return OptionalTensor(x);
}

template<>
inline
torch::optional<torch::ScalarType> from_ruby<torch::optional<torch::ScalarType>>(Object x)
{
  if (x.is_nil()) {
    return torch::nullopt;
  } else {
    return torch::optional<torch::ScalarType>{from_ruby<torch::ScalarType>(x)};
  }
}

template<>
inline
torch::optional<int64_t> from_ruby<torch::optional<int64_t>>(Object x)
{
  if (x.is_nil()) {
    return torch::nullopt;
  } else {
    return torch::optional<int64_t>{from_ruby<int64_t>(x)};
  }
}

template<>
inline
torch::optional<double> from_ruby<torch::optional<double>>(Object x)
{
  if (x.is_nil()) {
    return torch::nullopt;
  } else {
    return torch::optional<double>{from_ruby<double>(x)};
  }
}

template<>
inline
torch::optional<bool> from_ruby<torch::optional<bool>>(Object x)
{
  if (x.is_nil()) {
    return torch::nullopt;
  } else {
    return torch::optional<bool>{from_ruby<bool>(x)};
  }
}

template<>
inline
torch::optional<Scalar> from_ruby<torch::optional<Scalar>>(Object x)
{
  if (x.is_nil()) {
    return torch::nullopt;
  } else {
    return torch::optional<Scalar>{from_ruby<Scalar>(x)};
  }
}

Object wrap(std::tuple<torch::Tensor, torch::Tensor> x);
Object wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> x);
Object wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> x);
Object wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> x);
Object wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> x);
Object wrap(std::tuple<torch::Tensor, torch::Tensor, double, int64_t> x);
Object wrap(std::vector<torch::Tensor> x);
