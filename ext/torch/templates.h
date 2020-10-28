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
using torch::QScheme;
using torch::Generator;
using torch::TensorOptions;
using torch::Layout;
using torch::MemoryFormat;
using torch::IntArrayRef;
using torch::ArrayRef;
using torch::TensorList;
using torch::Storage;

#define HANDLE_TH_ERRORS                                             \
  try {

#define END_HANDLE_TH_ERRORS                                         \
  } catch (const torch::Error& ex) {                                 \
    rb_raise(rb_eRuntimeError, "%s", ex.what_without_backtrace());   \
  } catch (const Rice::Exception& ex) {                              \
    rb_raise(ex.class_of(), "%s", ex.what());                        \
  } catch (const std::exception& ex) {                               \
    rb_raise(rb_eRuntimeError, "%s", ex.what());                     \
  }

#define RETURN_NIL                                                   \
  return Qnil;

template<>
inline
std::vector<int64_t> from_ruby<std::vector<int64_t>>(Object x)
{
  Array a = Array(x);
  std::vector<int64_t> vec(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    vec[i] = from_ruby<int64_t>(a[i]);
  }
  return vec;
}

template<>
inline
std::vector<Tensor> from_ruby<std::vector<Tensor>>(Object x)
{
  Array a = Array(x);
  std::vector<Tensor> vec(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    vec[i] = from_ruby<Tensor>(a[i]);
  }
  return vec;
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

class OptionalTensor {
  torch::Tensor value;
  public:
    OptionalTensor(Object o) {
      if (o.is_nil()) {
        value = {};
      } else {
        value = from_ruby<torch::Tensor>(o);
      }
    }
    OptionalTensor(torch::Tensor o) {
      value = o;
    }
    operator torch::Tensor() const {
      return value;
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
