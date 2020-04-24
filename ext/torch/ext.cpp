#include <sstream>

#include <torch/torch.h>

#include <rice/Array.hpp>
#include <rice/Class.hpp>
#include <rice/Constructor.hpp>

#include "templates.hpp"

// generated with:
// rake generate:functions
#include "torch_functions.hpp"
#include "tensor_functions.hpp"
#include "nn_functions.hpp"

using namespace Rice;

// need to make a distinction between parameters and tensors
class Parameter: public torch::autograd::Variable {
  public:
    Parameter(Tensor&& t) : torch::autograd::Variable(t) { }
};

extern "C"
void Init_ext()
{
  Module rb_mTorch = define_module("Torch");
  add_torch_functions(rb_mTorch);

  Class rb_cTensor = define_class_under<torch::Tensor>(rb_mTorch, "Tensor");
  add_tensor_functions(rb_cTensor);

  Module rb_mNN = define_module_under(rb_mTorch, "NN");
  add_nn_functions(rb_mNN);

  rb_mTorch.define_singleton_method(
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
      "_save",
      *[](const Tensor &value) {
        auto v = torch::pickle_save(value);
        std::string str(v.begin(), v.end());
        return str;
      })
    .define_singleton_method(
      "_load",
      *[](const std::string &s) {
        std::vector<char> v;
        std::copy(s.begin(), s.end(), std::back_inserter(v));
        return torch::pickle_load(v).toTensor();
      })
    .define_singleton_method(
      "_binary_cross_entropy_with_logits",
      *[](const Tensor &input, const Tensor &target, OptionalTensor weight, OptionalTensor pos_weight, MyReduction reduction) {
        return torch::binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction);
      })
    .define_singleton_method(
      "_from_blob",
      *[](String s, IntArrayRef size, const torch::TensorOptions &options) {
        void *data = const_cast<char *>(s.c_str());
        return torch::from_blob(data, size, options);
      })
    .define_singleton_method(
      "_tensor",
      *[](Array a, IntArrayRef size, const torch::TensorOptions &options) {
        auto dtype = options.dtype();
        torch::Tensor t;
        if (dtype == torch::kBool) {
          std::vector<uint8_t> vec;
          for (size_t i = 0; i < a.size(); i++) {
            vec.push_back(from_ruby<bool>(a[i]));
          }
          t = torch::tensor(vec, options);
        } else {
          std::vector<float> vec;
          for (size_t i = 0; i < a.size(); i++) {
            vec.push_back(from_ruby<float>(a[i]));
          }
          // hack for requires_grad error
          if (options.requires_grad()) {
            t = torch::tensor(vec, options.requires_grad(c10::nullopt));
            t.set_requires_grad(true);
          } else {
            t = torch::tensor(vec, options);
          }
        }
        return t.reshape(size);
      });

  rb_cTensor
    .define_method("cuda?", &torch::Tensor::is_cuda)
    .define_method("sparse?", &torch::Tensor::is_sparse)
    .define_method("quantized?", &torch::Tensor::is_quantized)
    .define_method("dim", &torch::Tensor::dim)
    .define_method("numel", &torch::Tensor::numel)
    .define_method("element_size", &torch::Tensor::element_size)
    .define_method("requires_grad", &torch::Tensor::requires_grad)
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
      "_flat_data",
      *[](Tensor& self) {
        Tensor tensor = self;

        // move to CPU to get data
        if (tensor.device().type() != torch::kCPU) {
          torch::Device device("cpu");
          tensor = tensor.to(device);
        }

        Array a;
        auto dtype = tensor.dtype();

        // TODO DRY if someone knows C++
        if (dtype == torch::kByte) {
          uint8_t* data = tensor.data_ptr<uint8_t>();
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(data[i]);
          }
        } else if (dtype == torch::kChar) {
          int8_t* data = tensor.data_ptr<int8_t>();
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(to_ruby<int>(data[i]));
          }
        } else if (dtype == torch::kShort) {
          int16_t* data = tensor.data_ptr<int16_t>();
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(data[i]);
          }
        } else if (dtype == torch::kInt) {
          int32_t* data = tensor.data_ptr<int32_t>();
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(data[i]);
          }
        } else if (dtype == torch::kLong) {
          int64_t* data = tensor.data_ptr<int64_t>();
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(data[i]);
          }
        } else if (dtype == torch::kFloat) {
          float* data = tensor.data_ptr<float>();
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(data[i]);
          }
        } else if (dtype == torch::kDouble) {
          double* data = tensor.data_ptr<double>();
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(data[i]);
          }
        } else if (dtype == torch::kBool) {
          bool* data = tensor.data_ptr<bool>();
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(data[i] ? True : False);
          }
        } else {
          throw std::runtime_error("Unsupported type");
        }
        return a;
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
        return Parameter(std::move(var));
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
        try {
          // needed to catch exception
          torch::Device d(device);
          return self.device(d);
        } catch (const c10::Error& error) {
          throw std::runtime_error(error.what_without_backtrace());
        }
      })
    .define_method(
      "requires_grad",
      *[](torch::TensorOptions& self, bool requires_grad) {
        return self.requires_grad(requires_grad);
      });

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

  Class rb_cParameter = define_class_under<Parameter, torch::Tensor>(rb_mNN, "Parameter")
    .define_method(
      "grad",
      *[](Parameter& self) {
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
