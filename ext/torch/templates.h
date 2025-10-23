#pragma once

#include <string>

#ifdef isfinite
#undef isfinite
#endif

#include <rice/rice.hpp>

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
    rb_raise(rb_eTorchError, "%s", ex.what_without_backtrace());   \
  } catch (const Rice::Exception& ex) {                              \
    rb_raise(ex.class_of(), "%s", ex.what());                        \
  } catch (const std::exception& ex) {                               \
    rb_raise(rb_eRuntimeError, "%s", ex.what());                     \
  }

#define RETURN_NIL                                                   \
  return Qnil;

namespace Rice::detail {
  template<typename T>
  struct Type<c10::complex<T>> {
    static bool verify() { return true; }
  };

  template<typename T>
  class To_Ruby<c10::complex<T>> {
  public:
    explicit To_Ruby(Arg* arg) : arg_(arg) { }

    VALUE convert(c10::complex<T> const& x) {
      return rb_dbl_complex_new(x.real(), x.imag());
    }

  private:
    Arg* arg_ = nullptr;
  };

  template<typename T>
  class From_Ruby<c10::complex<T>> {
  public:
    From_Ruby() = default;

    explicit From_Ruby(Arg* arg) : arg_(arg) { }

    Convertible is_convertible(VALUE value) { return Convertible::Cast; }

    c10::complex<T> convert(VALUE x) {
      VALUE real = rb_funcall(x, rb_intern("real"), 0);
      VALUE imag = rb_funcall(x, rb_intern("imag"), 0);
      return c10::complex<T>(From_Ruby<T>().convert(real), From_Ruby<T>().convert(imag));
    }

  private:
    Arg* arg_ = nullptr;
  };

  template<>
  struct Type<FanModeType> {
    static bool verify() { return true; }
  };

  template<>
  class From_Ruby<FanModeType> {
  public:
    From_Ruby() = default;

    explicit From_Ruby(Arg* arg) : arg_(arg) { }

    Convertible is_convertible(VALUE value) { return Convertible::Cast; }

    FanModeType convert(VALUE x) {
      auto s = String(x).str();
      if (s == "fan_in") {
        return torch::kFanIn;
      } else if (s == "fan_out") {
        return torch::kFanOut;
      } else {
        throw std::runtime_error("Unsupported nonlinearity type: " + s);
      }
    }

  private:
    Arg* arg_ = nullptr;
  };

  template<>
  struct Type<NonlinearityType> {
    static bool verify() { return true; }
  };

  template<>
  class From_Ruby<NonlinearityType> {
  public:
    From_Ruby() = default;

    explicit From_Ruby(Arg* arg) : arg_(arg) { }

    Convertible is_convertible(VALUE value) { return Convertible::Cast; }

    NonlinearityType convert(VALUE x) {
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

  private:
    Arg* arg_ = nullptr;
  };

  template<>
  struct Type<Scalar> {
    static bool verify() { return true; }
  };

  template<>
  class From_Ruby<Scalar> {
  public:
    From_Ruby() = default;

    explicit From_Ruby(Arg* arg) : arg_(arg) { }

    Convertible is_convertible(VALUE value) { return Convertible::Cast; }

    Scalar convert(VALUE x) {
      if (FIXNUM_P(x)) {
        return torch::Scalar(From_Ruby<int64_t>().convert(x));
      } else {
        return torch::Scalar(From_Ruby<double>().convert(x));
      }
    }

  private:
    Arg* arg_ = nullptr;
  };
} // namespace Rice::detail
