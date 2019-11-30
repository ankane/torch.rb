#include <sstream>

#include <torch/torch.h>

#include <rice/Array.hpp>
#include <rice/Class.hpp>
#include <rice/Constructor.hpp>

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

extern "C"
void Init_ext()
{
  Module rb_mTorch = define_module("Torch")
    .define_singleton_method(
      "grad_enabled?",
      *[]() {
        return torch::GradMode::is_enabled();
      })
    .define_singleton_method(
      "_set_grad_enabled",
      *[](bool enabled) {
        torch::GradMode::set_enabled(enabled);
      })
    .define_singleton_method(
      "floating_point?",
      *[](torch::Tensor& input) {
        return torch::is_floating_point(input);
      })
    .define_singleton_method(
      "manual_seed",
      *[](uint64_t seed) {
        return torch::manual_seed(seed);
      })
    // begin tensor creation
    .define_singleton_method(
      "_arange",
      *[](Scalar start, Scalar end, Scalar step, const torch::TensorOptions &options) {
        return torch::arange(start, end, step, options);
      })
    .define_singleton_method(
      "_empty",
      *[](IntArrayRef size, const torch::TensorOptions &options) {
        return torch::empty(size, options);
      })
    .define_singleton_method(
      "_eye",
      *[](int64_t m, int64_t n, const torch::TensorOptions &options) {
        return torch::eye(m, n, options);
      })
    .define_singleton_method(
      "_full",
      *[](IntArrayRef size, Scalar fill_value, const torch::TensorOptions& options) {
        return torch::full(size, fill_value, options);
      })
    .define_singleton_method(
      "_linspace",
      *[](Scalar start, Scalar end, int64_t steps, const torch::TensorOptions& options) {
        return torch::linspace(start, end, steps, options);
      })
    .define_singleton_method(
      "_logspace",
      *[](Scalar start, Scalar end, int64_t steps, double base, const torch::TensorOptions& options) {
        return torch::logspace(start, end, steps, base, options);
      })
    .define_singleton_method(
      "_ones",
      *[](IntArrayRef size, const torch::TensorOptions &options) {
        return torch::ones(size, options);
      })
    .define_singleton_method(
      "_rand",
      *[](IntArrayRef size, const torch::TensorOptions &options) {
        return torch::rand(size, options);
      })
    .define_singleton_method(
      "_randint",
      *[](int64_t low, int64_t high, IntArrayRef size, const torch::TensorOptions &options) {
        return torch::randint(low, high, size, options);
      })
    .define_singleton_method(
      "_randn",
      *[](IntArrayRef size, const torch::TensorOptions &options) {
        return torch::randn(size, options);
      })
    .define_singleton_method(
      "_randperm",
      *[](int64_t n, const torch::TensorOptions &options) {
        return torch::randperm(n, options);
      })
    .define_singleton_method(
      "_zeros",
      *[](IntArrayRef size, const torch::TensorOptions &options) {
        return torch::zeros(size, options);
      })
    // begin operations
    .define_singleton_method(
      "_mean",
      *[](torch::Tensor& input) {
        return torch::mean(input);
      })
    .define_singleton_method(
      "_mean_dim",
      *[](torch::Tensor& input, int64_t dim, bool keepdim) {
        return torch::mean(input, dim, keepdim);
      })
    .define_singleton_method(
      "_sum",
      *[](torch::Tensor& input) {
        return torch::sum(input);
      })
    .define_singleton_method(
      "_sum_dim",
      *[](torch::Tensor& input, int64_t dim, bool keepdim) {
        return torch::sum(input, dim, keepdim);
      })
    .define_singleton_method(
      "_argmax",
      *[](torch::Tensor& input) {
        return torch::argmax(input);
      })
    .define_singleton_method(
      "_argmax_dim",
      *[](torch::Tensor& input, int64_t dim, bool keepdim) {
        return torch::argmax(input, dim, keepdim);
      })
    .define_singleton_method(
      "_cat",
      *[](TensorList tensors, int64_t dim) {
        return torch::cat(tensors, dim);
      })
    .define_singleton_method(
      "_norm",
      *[](torch::Tensor& input) {
        return torch::norm(input);
      })
    .define_singleton_method(
      "_min",
      *[](torch::Tensor& input) {
        return torch::min(input);
      })
    .define_singleton_method(
      "_max",
      *[](torch::Tensor& input) {
        return torch::max(input);
      })
    .define_singleton_method(
      "_max_out",
      *[](torch::Tensor &max, torch::Tensor &max_indices, const torch::Tensor &input, int64_t dim, bool keepdim) {
        // TODO add return value
        torch::_max_out(max, max_indices, input, dim, keepdim);
      })
    .define_singleton_method(
      "_sqrt",
      *[](torch::Tensor& input) {
        return torch::sqrt(input);
      })
    .define_singleton_method(
      "_exp",
      *[](torch::Tensor& input) {
        return torch::exp(input);
      })
    .define_singleton_method(
      "_log",
      *[](torch::Tensor& input) {
        return torch::log(input);
      })
    .define_singleton_method(
      "_sign",
      *[](torch::Tensor& input) {
        return torch::sign(input);
      })
    .define_singleton_method(
      "_unsqueeze",
      *[](torch::Tensor& input, int64_t dim) {
        return torch::unsqueeze(input, dim);
      })
    .define_singleton_method(
      "_dot",
      *[](torch::Tensor& input, torch::Tensor& tensor) {
        return torch::dot(input, tensor);
      })
    .define_singleton_method(
      "_matmul",
      *[](torch::Tensor& input, torch::Tensor& other) {
        return torch::matmul(input, other);
      })
    .define_singleton_method(
      "_eq",
      *[](torch::Tensor& input, torch::Tensor& other) {
        return torch::eq(input, other);
      })
    .define_singleton_method(
      "_gt",
      // TODO support tensors
      *[](torch::Tensor& input, Scalar other) {
        return torch::gt(input, other);
      })
    .define_singleton_method(
      "_lt",
      // TODO support tensors
      *[](torch::Tensor& input, Scalar other) {
        return torch::lt(input, other);
      })
    .define_singleton_method(
      "_add",
      *[](torch::Tensor& input, torch::Tensor& other) {
        return torch::add(input, other);
      })
    .define_singleton_method(
      "_add_scalar",
      *[](torch::Tensor& input, Scalar other) {
        return torch::add(input, other);
      })
    .define_singleton_method(
      "_add_out",
      *[](torch::Tensor& out, torch::Tensor& input, torch::Tensor& other) {
        return torch::add_out(out, input, other);
      })
    .define_singleton_method(
      "_sub",
      *[](torch::Tensor& input, torch::Tensor& other) {
        return torch::sub(input, other);
      })
    .define_singleton_method(
      "_sub_scalar",
      *[](torch::Tensor& input, Scalar other) {
        return torch::sub(input, other);
      })
    .define_singleton_method(
      "_mul",
      *[](torch::Tensor& input, torch::Tensor& other) {
        return torch::mul(input, other);
      })
    .define_singleton_method(
      "_mul_scalar",
      *[](torch::Tensor& input, Scalar other) {
        return torch::mul(input, other);
      })
    .define_singleton_method(
      "_div",
      *[](torch::Tensor& input, torch::Tensor& other) {
        return torch::div(input, other);
      })
    .define_singleton_method(
      "_div_scalar",
      *[](torch::Tensor& input, Scalar other) {
        return torch::div(input, other);
      })
    .define_singleton_method(
      "_remainder",
      *[](torch::Tensor& input, torch::Tensor& other) {
        return torch::remainder(input, other);
      })
    .define_singleton_method(
      "_remainder_scalar",
      *[](torch::Tensor& input, Scalar other) {
        return torch::remainder(input, other);
      })
    .define_singleton_method(
      "_pow",
      *[](torch::Tensor& input, Scalar exponent) {
        return torch::pow(input, exponent);
      })
    .define_singleton_method(
      "_abs",
      *[](torch::Tensor& input) {
        return torch::abs(input);
      })
    .define_singleton_method(
      "_neg",
      *[](torch::Tensor& input) {
        return torch::neg(input);
      })
    .define_singleton_method(
      "_reshape",
      *[](torch::Tensor& input, IntArrayRef shape) {
        return torch::reshape(input, shape);
      })
    .define_singleton_method(
      "_flatten",
      *[](torch::Tensor& input, int64_t start_dim, int64_t end_dim) {
        return torch::flatten(input, start_dim, end_dim);
      })
    .define_singleton_method(
      "relu",
      *[](torch::Tensor& input) {
        return torch::relu(input);
      })
    .define_singleton_method(
      "conv2d",
      *[](torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
        return torch::conv2d(input, weight, bias, stride, padding, dilation, groups);
      })
    .define_singleton_method(
      "linear",
      *[](torch::Tensor& input, torch::Tensor& weight, torch::Tensor& bias) {
        return torch::linear(input, weight, bias);
      })
    .define_singleton_method(
      "max_pool2d",
      *[](torch::Tensor& input, IntArrayRef kernel_size) {
        return torch::max_pool2d(input, kernel_size);
      })
    .define_singleton_method(
      "avg_pool2d",
      *[](torch::Tensor& input, IntArrayRef kernel_size) {
        return torch::avg_pool2d(input, kernel_size);
      })
    .define_singleton_method(
      "_dropout",
      *[](torch::Tensor& input, float p, bool train) {
        return torch::dropout(input, p, train);
      })
    .define_singleton_method(
      "_dropout!",
      *[](torch::Tensor& input, float p, bool train) {
        return torch::dropout_(input, p, train);
      })
    .define_singleton_method(
      "_feature_dropout",
      *[](torch::Tensor& input, float p, bool train) {
        return torch::feature_dropout(input, p, train);
      })
    .define_singleton_method(
      "_feature_dropout!",
      *[](torch::Tensor& input, float p, bool train) {
        return torch::feature_dropout_(input, p, train);
      })
    .define_singleton_method(
      "_alpha_dropout",
      *[](torch::Tensor& input, float p, bool train) {
        return torch::alpha_dropout(input, p, train);
      })
    .define_singleton_method(
      "_alpha_dropout!",
      *[](torch::Tensor& input, float p, bool train) {
        return torch::alpha_dropout_(input, p, train);
      })
    .define_singleton_method(
      "_feature_alpha_dropout",
      *[](torch::Tensor& input, float p, bool train) {
        return torch::feature_alpha_dropout(input, p, train);
      })
    .define_singleton_method(
      "_feature_alpha_dropout!",
      *[](torch::Tensor& input, float p, bool train) {
        return torch::feature_alpha_dropout_(input, p, train);
      })
    .define_singleton_method(
      "_embedding",
      // weight and indices are swapped from Python interface
      *[](const torch::Tensor &indices, const torch::Tensor &weight, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
        return torch::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
      })
    // loss functions
    .define_singleton_method(
      "l1_loss",
      *[](torch::Tensor& input, torch::Tensor& target, std::string reduction) {
        auto red = reduction == "mean" ? Reduction::Mean : Reduction::Sum;
        return torch::l1_loss(input, target, red);
      })
    .define_singleton_method(
      "mse_loss",
      *[](torch::Tensor& input, torch::Tensor& target, std::string reduction) {
        auto red = reduction == "mean" ? Reduction::Mean : Reduction::Sum;
        return torch::mse_loss(input, target, red);
      })
    .define_singleton_method(
      "nll_loss",
      *[](torch::Tensor& input, torch::Tensor& target, std::string reduction, int64_t ignore_index) {
        auto red = reduction == "mean" ? Reduction::Mean : Reduction::Sum;
        return torch::nll_loss(input, target, {}, red, ignore_index);
      })
    .define_singleton_method("numel", &torch::numel)
    .define_singleton_method(
      "_from_blob",
      *[](String s, IntArrayRef size, const torch::TensorOptions &options) {
        void *data = const_cast<char *>(s.c_str());
        return torch::from_blob(data, size, options);
      })
    .define_singleton_method(
      "_tensor",
      *[](Object o, IntArrayRef size, const torch::TensorOptions &options) {
        Array a = Array(o);
        std::vector<float> vec;
        for (size_t i = 0; i < a.size(); i++) {
          vec.push_back(from_ruby<float>(a[i]));
        }
        return torch::tensor(vec, options).reshape(size);
      });

  Class rb_cTensor = define_class_under<torch::Tensor>(rb_mTorch, "Tensor")
    .define_method("cuda?", &torch::Tensor::is_cuda)
    .define_method("distributed?", &torch::Tensor::is_distributed)
    .define_method("complex?", &torch::Tensor::is_complex)
    .define_method("floating_point?", &torch::Tensor::is_floating_point)
    .define_method("signed?", &torch::Tensor::is_signed)
    .define_method("sparse?", &torch::Tensor::is_sparse)
    .define_method("quantized?", &torch::Tensor::is_quantized)
    .define_method("dim", &torch::Tensor::dim)
    .define_method("element_size", &torch::Tensor::element_size)
    .define_method("requires_grad", &torch::Tensor::requires_grad)
    .define_method("view_as", &torch::Tensor::view_as)
    .define_method(
      "addcmul!",
      *[](torch::Tensor& self, Scalar value, const torch::Tensor & tensor1, const torch::Tensor & tensor2) {
        return self.addcmul_(tensor1, tensor2, value);
      })
    .define_method(
      "addcdiv!",
      *[](torch::Tensor& self, Scalar value, const torch::Tensor & tensor1, const torch::Tensor & tensor2) {
        return self.addcdiv_(tensor1, tensor2, value);
      })
    .define_method(
      "zero!",
      *[](torch::Tensor& self) {
        return self.zero_();
      })
    .define_method(
      "detach!",
      *[](torch::Tensor& self) {
        return self.detach_();
      })
    .define_method(
      "_select",
      *[](torch::Tensor& self, int64_t dim, int64_t index) {
        return self.select(dim, index);
      })
    .define_method(
      "_slice",
      *[](torch::Tensor& self, int64_t dim, int64_t start, int64_t end, int64_t step) {
        return self.slice(dim, start, end, step);
      })
    .define_method(
      "_requires_grad!",
      *[](torch::Tensor& self, bool requires_grad) {
        return self.set_requires_grad(requires_grad);
      })
    .define_method(
      "_backward",
      *[](torch::Tensor& self) {
        return self.backward();
      })
    .define_method(
      "_backward_gradient",
      *[](torch::Tensor& self, const torch::Tensor& gradient) {
        return self.backward(gradient);
      })
    .define_method(
      "grad",
      *[](torch::Tensor& self) {
        return self.grad();
      })
    .define_method(
      "_dtype",
      *[](torch::Tensor& self) {
        return (int) at::typeMetaToScalarType(self.dtype());
      })
    .define_method(
      "_type",
      *[](torch::Tensor& self, int dtype) {
        return self.toType((torch::ScalarType) dtype);
      })
    .define_method(
      "_layout",
      *[](torch::Tensor& self) {
        std::stringstream s;
        s << self.layout();
        return s.str();
      })
    .define_method(
      "device",
      *[](torch::Tensor& self) {
        std::stringstream s;
        s << self.device();
        return s.str();
      })
    .define_method(
      "_view",
      *[](torch::Tensor& self, IntArrayRef size) {
        return self.view(size);
      })
    .define_method(
      "resize_as!",
      *[](torch::Tensor& self, torch::Tensor& other) {
        return self.resize_as_(other);
      })
    .define_method(
      "fill!",
      *[](torch::Tensor& self, Scalar value) {
        return self.fill_(value);
      })
    .define_method(
      "_add!",
      *[](torch::Tensor& self, torch::Tensor& other) {
        return self.add_(other);
      })
    .define_method(
      "_add_alpha!",
      *[](torch::Tensor& self, torch::Tensor& other, Scalar alpha) {
        return self.add_(other, alpha);
      })
    .define_method(
      "_add_scalar!",
      *[](torch::Tensor& self, Scalar other) {
        return self.add_(other);
      })
    .define_method(
      "normal!",
      *[](torch::Tensor& self, double mean, double std) {
        return self.normal_(mean, std);
      })
    .define_method(
      "sub!",
      *[](torch::Tensor& self, torch::Tensor& other) {
        return self.sub_(other);
      })
    .define_method(
      "_mul!",
      *[](torch::Tensor& self, torch::Tensor& other) {
        return self.mul_(other);
      })
    .define_method(
      "_mul_scalar!",
      *[](torch::Tensor& self, Scalar other) {
        return self.mul_(other);
      })
    .define_method(
      "div!",
      *[](torch::Tensor& self, torch::Tensor& other) {
        return self.div_(other);
      })
    .define_method(
      "sqrt!",
      *[](torch::Tensor& self) {
        return self.sqrt_();
      })
    .define_method(
      "unsqueeze!",
      *[](torch::Tensor& self, int64_t dim) {
        return self.unsqueeze_(dim);
      })
    .define_method(
      "copy!",
      *[](torch::Tensor& self, torch::Tensor& src) {
        return self.copy_(src);
      })
    .define_method(
      "clone",
      *[](torch::Tensor& self) {
        return self.clone();
      })
    .define_method(
      "log_softmax",
      *[](torch::Tensor& self, int64_t dim) {
        return self.log_softmax(dim);
      })
    .define_method(
      "data",
      *[](torch::Tensor& self) {
        return self.data();
      })
    .define_method(
      "_data",
      *[](torch::Tensor& self) {
        Array a;
        auto dtype = self.dtype();

        // TODO DRY if someone knows C++
        if (dtype == torch::kByte) {
          uint8_t* data = self.data_ptr<uint8_t>();
          for (int i = 0; i < self.numel(); i++) {
            a.push(data[i]);
          }
        } else if (dtype == torch::kChar) {
          int8_t* data = self.data_ptr<int8_t>();
          for (int i = 0; i < self.numel(); i++) {
            a.push(to_ruby<int>(data[i]));
          }
        } else if (dtype == torch::kShort) {
          int16_t* data = self.data_ptr<int16_t>();
          for (int i = 0; i < self.numel(); i++) {
            a.push(data[i]);
          }
        } else if (dtype == torch::kInt) {
          int32_t* data = self.data_ptr<int32_t>();
          for (int i = 0; i < self.numel(); i++) {
            a.push(data[i]);
          }
        } else if (dtype == torch::kLong) {
          int64_t* data = self.data_ptr<int64_t>();
          for (int i = 0; i < self.numel(); i++) {
            a.push(data[i]);
          }
        } else if (dtype == torch::kFloat) {
          float* data = self.data_ptr<float>();
          for (int i = 0; i < self.numel(); i++) {
            a.push(data[i]);
          }
        } else if (dtype == torch::kDouble) {
          double* data = self.data_ptr<double>();
          for (int i = 0; i < self.numel(); i++) {
            a.push(data[i]);
          }
        } else if (dtype == torch::kBool) {
          bool* data = self.data_ptr<bool>();
          for (int i = 0; i < self.numel(); i++) {
            a.push(data[i] ? True : False);
          }
        } else {
          throw std::runtime_error("Unsupported type");
        }
        return a;
      })
    .define_method(
      "_size",
      *[](torch::Tensor& self, int i) {
        return self.size(i);
      })
    .define_method(
      "_to",
      *[](torch::Tensor& self, torch::Device device, int dtype, bool non_blocking, bool copy) {
        return self.to(device, (torch::ScalarType) dtype, non_blocking, copy);
      })
    .define_singleton_method(
      "_make_subclass",
      *[](torch::Tensor& rd, bool requires_grad) {
        auto data = torch::autograd::as_variable_ref(rd).detach();
        data.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
        auto var = data.set_requires_grad(requires_grad);
        return torch::autograd::Variable(std::move(var));
      });

  Class rb_cTensorOptions = define_class_under<torch::TensorOptions>(rb_mTorch, "TensorOptions")
    .define_constructor(Constructor<torch::TensorOptions>())
    .define_method(
      "dtype",
      *[](torch::TensorOptions& self, int dtype) {
        return self.dtype((torch::ScalarType) dtype);
      })
    .define_method(
      "layout",
      *[](torch::TensorOptions& self, std::string layout) {
        torch::Layout l;
        if (layout == "strided") {
          l = torch::kStrided;
        } else if (layout == "sparse") {
          l = torch::kSparse;
          throw std::runtime_error("Sparse layout not supported yet");
        } else {
          throw std::runtime_error("Unsupported layout: " + layout);
        }
        return self.layout(l);
      })
    .define_method(
      "device",
      *[](torch::TensorOptions& self, std::string device) {
        torch::DeviceType d;
        if (device == "cpu") {
          d = torch::kCPU;
        } else if (device == "cuda") {
          d = torch::kCUDA;
        } else {
          throw std::runtime_error("Unsupported device: " + device);
        }
        return self.device(d);
      })
    .define_method(
      "requires_grad",
      *[](torch::TensorOptions& self, bool requires_grad) {
        return self.requires_grad(requires_grad);
      });

  Module rb_mNN = define_module_under(rb_mTorch, "NN");

  Module rb_mInit = define_module_under(rb_mNN, "Init")
    .define_singleton_method(
      "kaiming_uniform!",
      *[](torch::Tensor& input, double a) {
        return torch::nn::init::kaiming_uniform_(input, a);
      })
    .define_singleton_method(
      "normal!",
      *[](torch::Tensor& input) {
        return torch::nn::init::normal_(input);
      })
    .define_singleton_method(
      "uniform!",
      *[](torch::Tensor& input, double to, double from) {
        return torch::nn::init::uniform_(input, to, from);
      });

  Class rb_cParameter = define_class_under<torch::autograd::Variable, torch::Tensor>(rb_mNN, "Parameter")
    // TODO return grad or nil to remove need for 2nd function
    .define_method(
      "_grad",
      *[](torch::autograd::Variable& self) {
        return self.grad();
      })
    .define_method(
      "_grad_defined",
      *[](torch::autograd::Variable& self) {
        return self.grad().defined();
      });

  Class rb_cDevice = define_class_under<torch::Device>(rb_mTorch, "Device")
    .define_constructor(Constructor<torch::Device, std::string>())
    .define_method("index", &torch::Device::index)
    .define_method("index?", &torch::Device::has_index)
    .define_method(
      "type",
      *[](torch::Device& self) {
        std::stringstream s;
        s << self.type();
        return s.str();
      });

  Module rb_mCUDA = define_module_under(rb_mTorch, "CUDA")
    .define_singleton_method("available?", &torch::cuda::is_available)
    .define_singleton_method("device_count", &torch::cuda::device_count);
}
