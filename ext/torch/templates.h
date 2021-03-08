#pragma once

#ifdef isfinite
#undef isfinite
#endif

#include <rice/rice.hpp>

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

namespace Rice::detail
{
  template<>
  struct From_Ruby<std::vector<int64_t>>
  {
    static std::vector<int64_t> convert(VALUE x)
    {
      Array a = Array(x);
      std::vector<int64_t> vec(a.size());
      for (size_t i = 0; i < a.size(); i++) {
        vec[i] = Rice::detail::From_Ruby<int64_t>::convert(a[i].value());
      }
      return vec;
    }
  };

  template<>
  struct From_Ruby<std::vector<Tensor>>
  {
    static std::vector<Tensor> convert(VALUE x)
    {
      Array a = Array(x);
      std::vector<Tensor> vec(a.size());
      for (size_t i = 0; i < a.size(); i++) {
        vec[i] = Rice::detail::From_Ruby<Tensor>::convert(a[i].value());
      }
      return vec;
    }
  };
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

namespace Rice::detail
{
  template<>
  struct From_Ruby<FanModeType>
  {
    static FanModeType convert(VALUE x)
    {
      return FanModeType(x);
    }
  };
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

namespace Rice::detail
{
  template<>
  struct From_Ruby<NonlinearityType>
  {
    static NonlinearityType convert(VALUE x)
    {
      return NonlinearityType(x);
    }
  };
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

namespace Rice::detail
{
  template<>
  struct From_Ruby<OptionalTensor>
  {
    static OptionalTensor convert(VALUE x)
    {
      return OptionalTensor(x);
    }
  };

  template<>
  struct From_Ruby<Scalar>
  {
    static Scalar convert(VALUE x)
    {
      if (FIXNUM_P(x)) {
        return torch::Scalar(Rice::detail::From_Ruby<int64_t>::convert(x));
      } else {
        return torch::Scalar(Rice::detail::From_Ruby<double>::convert(x));
      }
    }
  };

  template<>
  struct From_Ruby<torch::optional<torch::ScalarType>>
  {
    static torch::optional<torch::ScalarType> convert(VALUE x)
    {
      if (NIL_P(x)) {
        return torch::nullopt;
      } else {
        return torch::optional<torch::ScalarType>{Rice::detail::From_Ruby<torch::ScalarType>::convert(x)};
      }
    }
  };

  template<>
  struct From_Ruby<torch::optional<int64_t>>
  {
    static torch::optional<int64_t> convert(VALUE x)
    {
      if (NIL_P(x)) {
        return torch::nullopt;
      } else {
        return torch::optional<int64_t>{Rice::detail::From_Ruby<int64_t>::convert(x)};
      }
    }
  };

  template<>
  struct From_Ruby<torch::optional<double>>
  {
    static torch::optional<double> convert(VALUE x)
    {
      if (NIL_P(x)) {
        return torch::nullopt;
      } else {
        return torch::optional<double>{Rice::detail::From_Ruby<double>::convert(x)};
      }
    }
  };

  template<>
  struct From_Ruby<torch::optional<bool>>
  {
    static torch::optional<bool> convert(VALUE x)
    {
      if (NIL_P(x)) {
        return torch::nullopt;
      } else {
        return torch::optional<bool>{Rice::detail::From_Ruby<bool>::convert(x)};
      }
    }
  };

  template<>
  struct From_Ruby<torch::optional<Scalar>>
  {
    static torch::optional<Scalar> convert(VALUE x)
    {
      if (NIL_P(x)) {
        return torch::nullopt;
      } else {
        return torch::optional<Scalar>{Rice::detail::From_Ruby<Scalar>::convert(x)};
      }
    }
  };
}
