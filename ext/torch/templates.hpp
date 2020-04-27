#pragma once

#ifdef isfinite
#undef isfinite
#endif

#ifdef isinf
#undef isinf
#endif

#ifdef isnan
#undef isnan
#endif

#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#include <rice/Array.hpp>
#include <rice/Object.hpp>

using namespace Rice;

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

// for now
class Scalar {
  torch::Scalar value;
  public:
    Scalar(Object o) {
      // TODO cast based on Ruby type
      if (o.rb_type() == T_FIXNUM) {
        value = torch::Scalar(from_ruby<int64_t>(o));
      } else {
        value = torch::Scalar(from_ruby<float>(o));
      }
    }
    operator torch::Scalar() {
      return value;
    }
};

template<>
inline
Scalar from_ruby<Scalar>(Object x)
{
  return Scalar(x);
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

typedef torch::Tensor Tensor;

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
OptionalTensor from_ruby<OptionalTensor>(Object x)
{
  return OptionalTensor(x);
}

class ScalarType {
  Object value;
  public:
    ScalarType(Object o) {
      value = o;
    }
    operator at::ScalarType() {
      throw std::runtime_error("ScalarType arguments not implemented yet");
    }
};

template<>
inline
ScalarType from_ruby<ScalarType>(Object x)
{
  return ScalarType(x);
}

class OptionalScalarType {
  Object value;
  public:
    OptionalScalarType(Object o) {
      value = o;
    }
    operator c10::optional<at::ScalarType>() {
      if (value.is_nil()) {
        return c10::nullopt;
      }
      return ScalarType(value);
    }
};

template<>
inline
OptionalScalarType from_ruby<OptionalScalarType>(Object x)
{
  return OptionalScalarType(x);
}

typedef torch::Device Device;

Object wrap(std::tuple<torch::Tensor, torch::Tensor> x);
Object wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> x);
Object wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> x);
Object wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> x);
Object wrap(std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t> x);
Object wrap(std::tuple<torch::Tensor, torch::Tensor, double, int64_t> x);
