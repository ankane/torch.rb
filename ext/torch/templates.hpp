#pragma once

#include <rice/Array.hpp>
#include <rice/Object.hpp>

using namespace Rice;

template<>
inline
long long from_ruby<long long>(Object x)
{
  return NUM2LL(x);
}

template<>
inline
Object to_ruby<long long>(long long const & x)
{
  return LL2NUM(x);
}

template<>
inline
unsigned long long from_ruby<unsigned long long>(Object x)
{
  return NUM2ULL(x);
}

template<>
inline
Object to_ruby<unsigned long long>(unsigned long long const & x)
{
  return ULL2NUM(x);
}

template<>
inline
short from_ruby<short>(Object x)
{
  return NUM2SHORT(x);
}

template<>
inline
Object to_ruby<short>(short const & x)
{
  return INT2NUM(x);
}

template<>
inline
unsigned short from_ruby<unsigned short>(Object x)
{
  return NUM2USHORT(x);
}

template<>
inline
Object to_ruby<unsigned short>(unsigned short const & x)
{
  return UINT2NUM(x);
}

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
    // TODO switch NonlinearityType after LibTorch 1.4 release
    operator torch::nn::init::FanMode() {
      if (s == "fan_in") {
        return torch::nn::init::FanMode::FanIn;
      } else if (s == "fan_out") {
        return torch::nn::init::FanMode::FanOut;
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
    // TODO switch NonlinearityType after LibTorch 1.4 release
    operator torch::nn::init::Nonlinearity() {
      if (s == "linear") {
        return torch::nn::init::Nonlinearity::Linear;
      } else if (s == "conv1d") {
        return torch::nn::init::Nonlinearity::Conv1D;
      } else if (s == "conv2d") {
        return torch::nn::init::Nonlinearity::Conv2D;
      } else if (s == "conv3d") {
        return torch::nn::init::Nonlinearity::Conv3D;
      } else if (s == "conv_transpose1d") {
        return torch::nn::init::Nonlinearity::ConvTranspose1D;
      } else if (s == "conv_transpose2d") {
        return torch::nn::init::Nonlinearity::ConvTranspose2D;
      } else if (s == "conv_transpose3d") {
        return torch::nn::init::Nonlinearity::ConvTranspose3D;
      } else if (s == "sigmoid") {
        return torch::nn::init::Nonlinearity::Sigmoid;
      } else if (s == "tanh") {
        return torch::nn::init::Nonlinearity::Tanh;
      } else if (s == "relu") {
        return torch::nn::init::Nonlinearity::ReLU;
      } else if (s == "leaky_relu") {
        return torch::nn::init::Nonlinearity::LeakyReLU;
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
        return Reduction::None;
      }

      std::string s = String(value).str();
      if (s == "mean") {
        return Reduction::Mean;
      } else if (s == "sum") {
        return Reduction::Sum;
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
