#include <sstream>

#include <torch/torch.h>

#include <rice/Array.hpp>
#include <rice/Class.hpp>
#include <rice/Constructor.hpp>
#include <rice/Hash.hpp>

#include "templates.h"
#include "utils.h"

// generated with:
// rake generate:functions
#include "torch_functions.h"
#include "tensor_functions.h"
#include "nn_functions.h"

using namespace Rice;
using torch::indexing::TensorIndex;

// need to make a distinction between parameters and tensors
class Parameter: public torch::autograd::Variable {
  public:
    Parameter(Tensor&& t) : torch::autograd::Variable(t) { }
};

Class rb_cTensor;

std::vector<TensorIndex> index_vector(Array a) {
  Object obj;

  std::vector<TensorIndex> indices;
  indices.reserve(a.size());

  for (size_t i = 0; i < a.size(); i++) {
    obj = a[i];

    if (obj.is_instance_of(rb_cInteger)) {
      indices.push_back(from_ruby<int64_t>(obj));
    } else if (obj.is_instance_of(rb_cRange)) {
      torch::optional<int64_t> start_index = torch::nullopt;
      torch::optional<int64_t> stop_index = torch::nullopt;

      Object begin = obj.call("begin");
      if (!begin.is_nil()) {
        start_index = from_ruby<int64_t>(begin);
      }

      Object end = obj.call("end");
      if (!end.is_nil()) {
        stop_index = from_ruby<int64_t>(end);
      }

      Object exclude_end = obj.call("exclude_end?");
      if (stop_index.has_value() && !exclude_end) {
        if (stop_index.value() == -1) {
          stop_index = torch::nullopt;
        } else {
          stop_index = stop_index.value() + 1;
        }
      }

      indices.push_back(torch::indexing::Slice(start_index, stop_index));
    } else if (obj.is_instance_of(rb_cTensor)) {
      indices.push_back(from_ruby<Tensor>(obj));
    } else if (obj.is_nil()) {
      indices.push_back(torch::indexing::None);
    } else if (obj == True || obj == False) {
      indices.push_back(from_ruby<bool>(obj));
    } else {
      throw Exception(rb_eArgError, "Unsupported index type: %s", rb_obj_classname(obj));
    }
  }
  return indices;
}

void init_ivalue(Rice::Module& m);

extern "C"
void Init_ext()
{
  Module rb_mTorch = define_module("Torch");
  rb_mTorch.add_handler<torch::Error>(handle_error);
  add_torch_functions(rb_mTorch);

  rb_cTensor = define_class_under<torch::Tensor>(rb_mTorch, "Tensor");
  rb_cTensor.add_handler<torch::Error>(handle_error);
  add_tensor_functions(rb_cTensor);
  THPVariableClass = rb_cTensor.value();

  Module rb_mNN = define_module_under(rb_mTorch, "NN");
  rb_mNN.add_handler<torch::Error>(handle_error);
  add_nn_functions(rb_mNN);

  Module rb_mRandom = define_module_under(rb_mTorch, "Random")
    .add_handler<torch::Error>(handle_error)
    .define_singleton_method(
      "initial_seed",
      *[]() {
        return at::detail::getDefaultCPUGenerator().current_seed();
      })
    .define_singleton_method(
      "seed",
      *[]() {
        // TODO set for CUDA when available
        auto generator = at::detail::getDefaultCPUGenerator();
        return generator.seed();
      });

  init_ivalue(rb_mTorch);

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
    // config
    .define_singleton_method(
      "show_config",
      *[] {
        return torch::show_config();
      })
    .define_singleton_method(
      "parallel_info",
      *[] {
        return torch::get_parallel_info();
      })
    // begin operations
    .define_singleton_method(
      "_save",
      *[](const torch::IValue &value) {
        auto v = torch::pickle_save(value);
        std::string str(v.begin(), v.end());
        return str;
      })
    .define_singleton_method(
      "_load",
      *[](const std::string &s) {
        std::vector<char> v;
        std::copy(s.begin(), s.end(), std::back_inserter(v));
        // https://github.com/pytorch/pytorch/issues/20356#issuecomment-567663701
        return torch::pickle_load(v);
      })
    .define_singleton_method(
      "_from_blob",
      *[](String s, std::vector<int64_t> size, const torch::TensorOptions &options) {
        void *data = const_cast<char *>(s.c_str());
        return torch::from_blob(data, size, options);
      })
    .define_singleton_method(
      "_tensor",
      *[](Array a, std::vector<int64_t> size, const torch::TensorOptions &options) {
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
    // in C++ for performance
    .define_method(
      "shape",
      *[](Tensor& self) {
        Array a;
        for (auto &size : self.sizes()) {
          a.push(size);
        }
        return a;
      })
    .define_method(
      "_strides",
      *[](Tensor& self) {
        Array a;
        for (auto &stride : self.strides()) {
          a.push(stride);
        }
        return a;
      })
    .define_method(
      "_index",
      *[](Tensor& self, Array indices) {
        auto vec = index_vector(indices);
        return self.index(vec);
      })
    .define_method(
      "_index_put_custom",
      *[](Tensor& self, Array indices, torch::Tensor& value) {
        auto vec = index_vector(indices);
        return self.index_put_(vec, value);
      })
    .define_method(
      "contiguous?",
      *[](Tensor& self) {
        return self.is_contiguous();
      })
    .define_method(
      "_requires_grad!",
      *[](Tensor& self, bool requires_grad) {
        return self.set_requires_grad(requires_grad);
      })
    .define_method(
      "grad",
      *[](Tensor& self) {
        auto grad = self.grad();
        return grad.defined() ? to_ruby<torch::Tensor>(grad) : Nil;
      })
    .define_method(
      "grad=",
      *[](Tensor& self, torch::Tensor& grad) {
        self.mutable_grad() = grad;
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
      "_data_str",
      *[](Tensor& self) {
        Tensor tensor = self;

        // move to CPU to get data
        if (tensor.device().type() != torch::kCPU) {
          torch::Device device("cpu");
          tensor = tensor.to(device);
        }

        if (!tensor.is_contiguous()) {
          tensor = tensor.contiguous();
        }

        auto data_ptr = (const char *) tensor.data_ptr();
        return std::string(data_ptr, tensor.numel() * tensor.element_size());
      })
    // for TorchVision
    .define_method(
      "_data_ptr",
      *[](Tensor& self) {
        return reinterpret_cast<uintptr_t>(self.data_ptr());
      })
    // TODO figure out a better way to do this
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

        Tensor view = tensor.reshape({tensor.numel()});

        // TODO DRY if someone knows C++
        if (dtype == torch::kByte) {
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(view[i].item().to<uint8_t>());
          }
        } else if (dtype == torch::kChar) {
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(to_ruby<int>(view[i].item().to<int8_t>()));
          }
        } else if (dtype == torch::kShort) {
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(view[i].item().to<int16_t>());
          }
        } else if (dtype == torch::kInt) {
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(view[i].item().to<int32_t>());
          }
        } else if (dtype == torch::kLong) {
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(view[i].item().to<int64_t>());
          }
        } else if (dtype == torch::kFloat) {
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(view[i].item().to<float>());
          }
        } else if (dtype == torch::kDouble) {
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(view[i].item().to<double>());
          }
        } else if (dtype == torch::kBool) {
          for (int i = 0; i < tensor.numel(); i++) {
            a.push(view[i].item().to<bool>() ? True : False);
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
        auto data = rd.detach();
        data.unsafeGetTensorImpl()->set_allow_tensor_metadata_change(true);
        auto var = data.set_requires_grad(requires_grad);
        return Parameter(std::move(var));
      });

  Class rb_cTensorOptions = define_class_under<torch::TensorOptions>(rb_mTorch, "TensorOptions")
    .add_handler<torch::Error>(handle_error)
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
        torch::Device d(device);
        return self.device(d);
      })
    .define_method(
      "requires_grad",
      *[](torch::TensorOptions& self, bool requires_grad) {
        return self.requires_grad(requires_grad);
      });

  Module rb_mInit = define_module_under(rb_mNN, "Init")
    .add_handler<torch::Error>(handle_error)
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
    .add_handler<torch::Error>(handle_error)
    .define_method(
      "grad",
      *[](Parameter& self) {
        auto grad = self.grad();
        return grad.defined() ? to_ruby<torch::Tensor>(grad) : Nil;
      })
    .define_method(
      "grad=",
      *[](Parameter& self, torch::Tensor& grad) {
        self.mutable_grad() = grad;
      });

  Class rb_cDevice = define_class_under<torch::Device>(rb_mTorch, "Device")
    .add_handler<torch::Error>(handle_error)
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
    .add_handler<torch::Error>(handle_error)
    .define_singleton_method("available?", &torch::cuda::is_available)
    .define_singleton_method("device_count", &torch::cuda::device_count)
    .define_singleton_method("manual_seed", &torch::cuda::manual_seed)
    .define_singleton_method("manual_seed_all", &torch::cuda::manual_seed_all);
}
