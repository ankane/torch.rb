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
using ScalarList = ArrayRef<Scalar>;

using torch::nn::init::FanModeType;
using torch::nn::init::NonlinearityType;

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

class OptionalTensor {
  torch::Tensor value;
  public:
    OptionalTensor(Object o) {
      if (o.is_nil()) {
        value = torch::Tensor();
      } else {
        value = Rice::detail::From_Ruby<torch::Tensor>().convert(o.value());
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
  struct Type<FanModeType>
  {
    static bool verify()
    {
      return true;
    }
  };

  template<>
  class From_Ruby<FanModeType>
  {
  public:
    FanModeType convert(VALUE x)
    {
      auto s = String(x).str();
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
  struct Type<NonlinearityType>
  {
    static bool verify()
    {
      return true;
    }
  };

  template<>
  class From_Ruby<NonlinearityType>
  {
  public:
    NonlinearityType convert(VALUE x)
    {
      auto s = String(x).str();
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
  struct Type<OptionalTensor>
  {
    static bool verify()
    {
      return true;
    }
  };

  template<>
  class From_Ruby<OptionalTensor>
  {
  public:
    OptionalTensor convert(VALUE x)
    {
      return OptionalTensor(x);
    }
  };

  template<>
  struct Type<Scalar>
  {
    static bool verify()
    {
      return true;
    }
  };

  template<>
  class From_Ruby<Scalar>
  {
  public:
    Scalar convert(VALUE x)
    {
      if (FIXNUM_P(x)) {
        return torch::Scalar(From_Ruby<int64_t>().convert(x));
      } else {
        return torch::Scalar(From_Ruby<double>().convert(x));
      }
    }
  };

  template<typename T>
  struct Type<torch::optional<T>>
  {
    static bool verify()
    {
      return true;
    }
  };

  template<typename T>
  class From_Ruby<torch::optional<T>>
  {
  public:
    torch::optional<T> convert(VALUE x)
    {
      if (NIL_P(x)) {
        return torch::nullopt;
      } else {
        return torch::optional<T>{From_Ruby<T>().convert(x)};
      }
    }
  };
}
