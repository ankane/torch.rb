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

Object tensor_array(std::tuple<torch::Tensor, torch::Tensor> x) {
  Array a;
  a.push(to_ruby<torch::Tensor>(std::get<0>(x)));
  a.push(to_ruby<torch::Tensor>(std::get<1>(x)));
  return Object(a);
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
      *[](Tensor& input) {
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
      *[](Tensor& input) {
        return torch::mean(input);
      })
    .define_singleton_method(
      "_mean_dim",
      *[](Tensor& input, int64_t dim, bool keepdim) {
        return torch::mean(input, dim, keepdim);
      })
    .define_singleton_method(
      "_sum",
      *[](Tensor& input) {
        return torch::sum(input);
      })
    .define_singleton_method(
      "_sum_dim",
      *[](Tensor& input, int64_t dim, bool keepdim) {
        return torch::sum(input, dim, keepdim);
      })
    .define_singleton_method(
      "_argmax",
      *[](Tensor& input) {
        return torch::argmax(input);
      })
    .define_singleton_method(
      "_argmax_dim",
      *[](Tensor& input, int64_t dim, bool keepdim) {
        return torch::argmax(input, dim, keepdim);
      })
    .define_singleton_method(
      "_cat",
      *[](TensorList tensors, int64_t dim) {
        return torch::cat(tensors, dim);
      })
    .define_singleton_method(
      "_norm",
      *[](Tensor& input) {
        return torch::norm(input);
      })
    .define_singleton_method(
      "_min",
      *[](Tensor& input) {
        return torch::min(input);
      })
    .define_singleton_method(
      "_max",
      *[](Tensor& input) {
        return torch::max(input);
      })
    .define_singleton_method(
      "_max_out",
      *[](Tensor &max, Tensor &max_indices, const Tensor &input, int64_t dim, bool keepdim) {
        return tensor_array(torch::_max_out(max, max_indices, input, dim, keepdim));
      })
    .define_singleton_method(
      "_sqrt",
      *[](Tensor& input) {
        return torch::sqrt(input);
      })
    .define_singleton_method(
      "_exp",
      *[](Tensor& input) {
        return torch::exp(input);
      })
    .define_singleton_method(
      "_log",
      *[](Tensor& input) {
        return torch::log(input);
      })
    .define_singleton_method(
      "_sign",
      *[](Tensor& input) {
        return torch::sign(input);
      })
    .define_singleton_method(
      "_unsqueeze",
      *[](Tensor& input, int64_t dim) {
        return torch::unsqueeze(input, dim);
      })
    .define_singleton_method(
      "_dot",
      *[](Tensor& input, Tensor& tensor) {
        return torch::dot(input, tensor);
      })
    .define_singleton_method(
      "_matmul",
      *[](Tensor& input, Tensor& other) {
        return torch::matmul(input, other);
      })
    .define_singleton_method(
      "_eq",
      *[](Tensor& input, Tensor& other) {
        return torch::eq(input, other);
      })
    .define_singleton_method(
      "_gt",
      // TODO support tensors
      *[](Tensor& input, Scalar other) {
        return torch::gt(input, other);
      })
    .define_singleton_method(
      "_lt",
      // TODO support tensors
      *[](Tensor& input, Scalar other) {
        return torch::lt(input, other);
      })
    .define_singleton_method(
      "_add",
      *[](Tensor& input, Tensor& other) {
        return torch::add(input, other);
      })
    .define_singleton_method(
      "_add_scalar",
      *[](Tensor& input, Scalar other) {
        return torch::add(input, other);
      })
    .define_singleton_method(
      "_add_out",
      *[](Tensor& out, Tensor& input, Tensor& other) {
        return torch::add_out(out, input, other);
      })
    .define_singleton_method(
      "_sub",
      *[](Tensor& input, Tensor& other) {
        return torch::sub(input, other);
      })
    .define_singleton_method(
      "_sub_scalar",
      *[](Tensor& input, Scalar other) {
        return torch::sub(input, other);
      })
    .define_singleton_method(
      "_mul",
      *[](Tensor& input, Tensor& other) {
        return torch::mul(input, other);
      })
    .define_singleton_method(
      "_mul_scalar",
      *[](Tensor& input, Scalar other) {
        return torch::mul(input, other);
      })
    .define_singleton_method(
      "_div",
      *[](Tensor& input, Tensor& other) {
        return torch::div(input, other);
      })
    .define_singleton_method(
      "_div_scalar",
      *[](Tensor& input, Scalar other) {
        return torch::div(input, other);
      })
    .define_singleton_method(
      "_remainder",
      *[](Tensor& input, Tensor& other) {
        return torch::remainder(input, other);
      })
    .define_singleton_method(
      "_remainder_scalar",
      *[](Tensor& input, Scalar other) {
        return torch::remainder(input, other);
      })
    .define_singleton_method(
      "_pow",
      *[](Tensor& input, Scalar exponent) {
        return torch::pow(input, exponent);
      })
    .define_singleton_method(
      "_topk",
      *[](Tensor& input, int64_t k) {
        return tensor_array(torch::topk(input, k));
      })
    .define_singleton_method(
      "_sigmoid",
      *[](Tensor& input) {
        return torch::sigmoid(input);
      })
    .define_singleton_method(
      "_softplus",
      *[](const Tensor &input, Scalar beta, Scalar threshold) {
        return torch::softplus(input, beta, threshold);
      })
    .define_singleton_method(
      "_softmax",
      *[](const Tensor &input, int64_t dim) {
        return torch::softmax(input, dim);
      })
    .define_singleton_method(
      "_log_softmax",
      *[](Tensor& input, int64_t dim) {
        return torch::log_softmax(input, dim);
      })
    .define_singleton_method(
      "_abs",
      *[](Tensor& input) {
        return torch::abs(input);
      })
    .define_singleton_method(
      "_neg",
      *[](Tensor& input) {
        return torch::neg(input);
      })
    .define_singleton_method(
      "_reshape",
      *[](Tensor& input, IntArrayRef shape) {
        return torch::reshape(input, shape);
      })
    .define_singleton_method(
      "_flatten",
      *[](Tensor& input, int64_t start_dim, int64_t end_dim) {
        return torch::flatten(input, start_dim, end_dim);
      })
    .define_singleton_method(
      "relu",
      *[](Tensor& input) {
        return torch::relu(input);
      })
    .define_singleton_method(
      "prelu",
      *[](torch::Tensor& input, torch::Tensor& weight) {
        return torch::prelu(input, weight);
      })
    .define_singleton_method(
      "leaky_relu",
      *[](torch::Tensor& input, Scalar negative_slope) {
        return torch::leaky_relu(input, negative_slope);
      })
    .define_singleton_method(
      "conv2d",
      *[](Tensor& input, Tensor& weight, Tensor& bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups) {
        return torch::conv2d(input, weight, bias, stride, padding, dilation, groups);
      })
    // linear layers
    .define_singleton_method(
      "bilinear",
      *[](const Tensor &input1, const Tensor &input2, const Tensor &weight, const Tensor &bias) {
        return torch::bilinear(input1, input2, weight, bias);
      })
    .define_singleton_method(
      "linear",
      *[](Tensor& input, Tensor& weight, Tensor& bias) {
        return torch::linear(input, weight, bias);
      })
    // pooling layers
    .define_singleton_method(
      "max_pool2d",
      *[](Tensor& input, IntArrayRef kernel_size) {
        return torch::max_pool2d(input, kernel_size);
      })
    .define_singleton_method(
      "avg_pool2d",
      *[](Tensor& input, IntArrayRef kernel_size) {
        return torch::avg_pool2d(input, kernel_size);
      })
    .define_singleton_method(
      "_dropout",
      *[](Tensor& input, float p, bool train) {
        return torch::dropout(input, p, train);
      })
    .define_singleton_method(
      "_dropout!",
      *[](Tensor& input, float p, bool train) {
        return torch::dropout_(input, p, train);
      })
    .define_singleton_method(
      "_feature_dropout",
      *[](Tensor& input, float p, bool train) {
        return torch::feature_dropout(input, p, train);
      })
    .define_singleton_method(
      "_feature_dropout!",
      *[](Tensor& input, float p, bool train) {
        return torch::feature_dropout_(input, p, train);
      })
    .define_singleton_method(
      "_alpha_dropout",
      *[](Tensor& input, float p, bool train) {
        return torch::alpha_dropout(input, p, train);
      })
    .define_singleton_method(
      "_alpha_dropout!",
      *[](Tensor& input, float p, bool train) {
        return torch::alpha_dropout_(input, p, train);
      })
    .define_singleton_method(
      "_feature_alpha_dropout",
      *[](Tensor& input, float p, bool train) {
        return torch::feature_alpha_dropout(input, p, train);
      })
    .define_singleton_method(
      "_feature_alpha_dropout!",
      *[](Tensor& input, float p, bool train) {
        return torch::feature_alpha_dropout_(input, p, train);
      })
    // sparse layers
    .define_singleton_method(
      "_embedding",
      // weight and indices are swapped from Python interface
      *[](const Tensor &indices, const Tensor &weight, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
        return torch::embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
      })
    .define_singleton_method(
      "_embedding_bag",
      // weight and indices are swapped from Python interface
      *[](const Tensor &weight, const Tensor &indices, const Tensor &offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor &per_sample_weights) {
        return torch::embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights);
      })
    // distance functions
    .define_singleton_method(
      "_cosine_similarity",
      *[](const Tensor &x1, const Tensor &x2, int64_t dim, double eps) {
        return torch::cosine_similarity(x1, x2, dim, eps);
      })
    .define_singleton_method(
      "_pairwise_distance",
      *[](const Tensor &x1, const Tensor &x2, double p, double eps, bool keepdim) {
        return torch::pairwise_distance(x1, x2, p, eps, keepdim);
      })
    // loss functions
    .define_singleton_method(
      "binary_cross_entropy",
      *[](Tensor& input, Tensor& target, MyReduction reduction) {
        return torch::binary_cross_entropy(input, target, {}, reduction);
      })
    .define_singleton_method(
      "ctc_loss",
      *[](const Tensor &log_probs, const Tensor &targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, MyReduction reduction, bool zero_infinity) {
        return torch::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, reduction, zero_infinity);
      })
    .define_singleton_method(
      "hinge_embedding_loss",
      *[](const Tensor &input, const Tensor &target, double margin, MyReduction reduction) {
        return torch::hinge_embedding_loss(input, target, margin, reduction);
      })
    .define_singleton_method(
      "kl_div",
      *[](Tensor& input, Tensor& target, MyReduction reduction) {
        return torch::kl_div(input, target, reduction);
      })
    .define_singleton_method(
      "l1_loss",
      *[](Tensor& input, Tensor& target, MyReduction reduction) {
        return torch::l1_loss(input, target, reduction);
      })
    .define_singleton_method(
      "mse_loss",
      *[](Tensor& input, Tensor& target, MyReduction reduction) {
        return torch::mse_loss(input, target, reduction);
      })
    .define_singleton_method(
      "multilabel_margin_loss",
      *[](const Tensor &input, const Tensor &target, MyReduction reduction) {
        return torch::multilabel_margin_loss(input, target, reduction);
      })
    .define_singleton_method(
      "nll_loss",
      *[](Tensor& input, Tensor& target, MyReduction reduction, int64_t ignore_index) {
        return torch::nll_loss(input, target, {}, reduction, ignore_index);
      })
    .define_singleton_method(
      "poisson_nll_loss",
      *[](const Tensor &input, const Tensor &target, bool log_input, bool full, double eps, MyReduction reduction) {
        return torch::poisson_nll_loss(input, target, log_input, full, eps, reduction);
      })
    .define_singleton_method(
      "soft_margin_loss",
      *[](const Tensor &input, const Tensor &target, MyReduction reduction) {
        return torch::soft_margin_loss(input, target, reduction);
      })
    .define_singleton_method(
      "smooth_l1_loss",
      *[](const Tensor &input, const Tensor &target, MyReduction reduction) {
        return torch::smooth_l1_loss(input, target, reduction);
      })
    // end loss
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
        auto dtype = options.dtype();
        torch::Tensor t;
        if (dtype == torch::kBool) {
          throw std::runtime_error("Cannot create bool from tensor method yet");
        } else {
          std::vector<float> vec;
          for (size_t i = 0; i < a.size(); i++) {
            vec.push_back(from_ruby<float>(a[i]));
          }
          t = torch::tensor(vec, options);
        }
        return t.reshape(size);
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
      *[](Tensor& self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {
        return self.addcmul_(tensor1, tensor2, value);
      })
    .define_method(
      "addcdiv!",
      *[](Tensor& self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {
        return self.addcdiv_(tensor1, tensor2, value);
      })
    .define_method(
      "zero!",
      *[](Tensor& self) {
        return self.zero_();
      })
    .define_method(
      "detach",
      *[](Tensor& self) {
        return self.detach();
      })
    .define_method(
      "detach!",
      *[](Tensor& self) {
        return self.detach_();
      })
    .define_method(
      "_select",
      *[](Tensor& self, int64_t dim, int64_t index) {
        return self.select(dim, index);
      })
    .define_method(
      "_slice",
      *[](Tensor& self, int64_t dim, int64_t start, int64_t end, int64_t step) {
        return self.slice(dim, start, end, step);
      })
    .define_method(
      "_requires_grad!",
      *[](Tensor& self, bool requires_grad) {
        return self.set_requires_grad(requires_grad);
      })
    .define_method(
      "_backward",
      *[](Tensor& self, Object gradient) {
        return gradient.is_nil() ? self.backward() : self.backward(from_ruby<torch::Tensor>(gradient));
      })
    .define_method(
      "grad",
      *[](Tensor& self) {
        return self.grad();
      })
    .define_method(
      "_dtype",
      *[](Tensor& self) {
        return (int) at::typeMetaToScalarType(self.dtype());
      })
    .define_method(
      "_type",
      *[](Tensor& self, int dtype) {
        return self.toType((torch::ScalarType) dtype);
      })
    .define_method(
      "_layout",
      *[](Tensor& self) {
        std::stringstream s;
        s << self.layout();
        return s.str();
      })
    .define_method(
      "device",
      *[](Tensor& self) {
        std::stringstream s;
        s << self.device();
        return s.str();
      })
    .define_method(
      "_view",
      *[](Tensor& self, IntArrayRef size) {
        return self.view(size);
      })
    .define_method(
      "resize_as!",
      *[](Tensor& self, Tensor& other) {
        return self.resize_as_(other);
      })
    .define_method(
      "fill!",
      *[](Tensor& self, Scalar value) {
        return self.fill_(value);
      })
    .define_method(
      "relu!",
      *[](Tensor& self) {
        return self.relu_();
      })
    .define_method(
      "_add!",
      *[](Tensor& self, Tensor& other) {
        return self.add_(other);
      })
    .define_method(
      "_add_alpha!",
      *[](Tensor& self, Tensor& other, Scalar alpha) {
        return self.add_(other, alpha);
      })
    .define_method(
      "_add_scalar!",
      *[](Tensor& self, Scalar other) {
        return self.add_(other);
      })
    .define_method(
      "normal!",
      *[](Tensor& self, double mean, double std) {
        return self.normal_(mean, std);
      })
    .define_method(
      "random!",
      *[](Tensor& self, int64_t to) {
        return self.random_(to);
      })
    .define_method(
      "sub!",
      *[](Tensor& self, Tensor& other) {
        return self.sub_(other);
      })
    .define_method(
      "_mul!",
      *[](Tensor& self, Tensor& other) {
        return self.mul_(other);
      })
    .define_method(
      "_mul_scalar!",
      *[](Tensor& self, Scalar other) {
        return self.mul_(other);
      })
    .define_method(
      "div!",
      *[](Tensor& self, Tensor& other) {
        return self.div_(other);
      })
    .define_method(
      "sqrt!",
      *[](Tensor& self) {
        return self.sqrt_();
      })
    .define_method(
      "unsqueeze!",
      *[](Tensor& self, int64_t dim) {
        return self.unsqueeze_(dim);
      })
    .define_method(
      "copy!",
      *[](Tensor& self, Tensor& src) {
        return self.copy_(src);
      })
    .define_method(
      "clone",
      *[](Tensor& self) {
        return self.clone();
      })
    .define_method(
      "data",
      *[](Tensor& self) {
        return self.data();
      })
    .define_method(
      "_data",
      *[](Tensor& self) {
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
      *[](Tensor& self, int i) {
        return self.size(i);
      })
    .define_method(
      "_to",
      *[](Tensor& self, torch::Device device, int dtype, bool non_blocking, bool copy) {
        return self.to(device, (torch::ScalarType) dtype, non_blocking, copy);
      })
    .define_singleton_method(
      "_make_subclass",
      *[](Tensor& rd, bool requires_grad) {
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
      "_calculate_gain",
      *[](NonlinearityType nonlinearity, double param) {
        return torch::nn::init::calculate_gain(nonlinearity, param);
      })
    .define_singleton_method(
      "_uniform!",
      *[](Tensor tensor, double low, double high) {
        return torch::nn::init::uniform_(tensor, low, high);
      })
    .define_singleton_method(
      "_normal!",
      *[](Tensor tensor, double mean, double std) {
        return torch::nn::init::normal_(tensor, mean, std);
      })
    .define_singleton_method(
      "_constant!",
      *[](Tensor tensor, Scalar value) {
        return torch::nn::init::constant_(tensor, value);
      })
    .define_singleton_method(
      "_ones!",
      *[](Tensor tensor) {
        return torch::nn::init::ones_(tensor);
      })
    .define_singleton_method(
      "_zeros!",
      *[](Tensor tensor) {
        return torch::nn::init::zeros_(tensor);
      })
    .define_singleton_method(
      "_eye!",
      *[](Tensor tensor) {
        return torch::nn::init::eye_(tensor);
      })
    .define_singleton_method(
      "_dirac!",
      *[](Tensor tensor) {
        return torch::nn::init::dirac_(tensor);
      })
    .define_singleton_method(
      "_xavier_uniform!",
      *[](Tensor tensor, double gain) {
        return torch::nn::init::xavier_uniform_(tensor, gain);
      })
    .define_singleton_method(
      "_xavier_normal!",
      *[](Tensor tensor, double gain) {
        return torch::nn::init::xavier_normal_(tensor, gain);
      })
    .define_singleton_method(
      "_kaiming_uniform!",
      *[](Tensor tensor, double a, FanModeType mode, NonlinearityType nonlinearity) {
        return torch::nn::init::kaiming_uniform_(tensor, a, mode, nonlinearity);
      })
    .define_singleton_method(
      "_kaiming_normal!",
      *[](Tensor tensor, double a, FanModeType mode, NonlinearityType nonlinearity) {
        return torch::nn::init::kaiming_normal_(tensor, a, mode, nonlinearity);
      })
    .define_singleton_method(
      "_orthogonal!",
      *[](Tensor tensor, double gain) {
        return torch::nn::init::orthogonal_(tensor, gain);
      })
    .define_singleton_method(
      "_sparse!",
      *[](Tensor tensor, double sparsity, double std) {
        return torch::nn::init::sparse_(tensor, sparsity, std);
      });

  Class rb_cParameter = define_class_under<torch::autograd::Variable, torch::Tensor>(rb_mNN, "Parameter")
    .define_method(
      "grad",
      *[](torch::autograd::Variable& self) {
        auto grad = self.grad();
        return grad.defined() ? to_ruby<torch::Tensor>(grad) : Nil;
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
