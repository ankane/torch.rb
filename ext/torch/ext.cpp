#include <sstream>

#include <torch/torch.h>

#include <rice/Array.hpp>
#include <rice/Class.hpp>
#include <rice/Constructor.hpp>
#include <rice/Hash.hpp>

#include "templates.hpp"

// generated with:
// rake generate:functions
#include "torch_functions.hpp"
#include "tensor_functions.hpp"
#include "nn_functions.hpp"

using namespace Rice;
using torch::indexing::TensorIndex;

// need to make a distinction between parameters and tensors
class Parameter: public torch::autograd::Variable {
  public:
    Parameter(Tensor&& t) : torch::autograd::Variable(t) { }
};

void handle_error(torch::Error const & ex)
{
  throw Exception(rb_eRuntimeError, ex.what_without_backtrace());
}

Class rb_cTensor;

std::vector<TensorIndex> index_vector(Array a) {
  Object obj;

  std::vector<TensorIndex> indices;
  indices.reserve(a.size());

  for (size_t i = 0; i < a.size(); i++) {
    obj = a[i];

    if (obj.is_instance_of(rb_cInteger)) {
      indices.push_back(TensorIndex(from_ruby<int64_t>(obj)));
    } else if (obj.is_instance_of(rb_cRange)) {
      torch::optional<int64_t> start_index = from_ruby<int64_t>(obj.call("begin"));
      torch::optional<int64_t> stop_index = -1;

      Object end = obj.call("end");
      if (!end.is_nil()) {
        stop_index = from_ruby<int64_t>(end);
      }

      Object exclude_end = obj.call("exclude_end?");
      if (!exclude_end) {
        if (stop_index.value() == -1) {
          stop_index = torch::nullopt;
        } else {
          stop_index = stop_index.value() + 1;
        }
      }

      indices.push_back(TensorIndex(torch::indexing::Slice(start_index, stop_index)));
    } else if (obj.is_instance_of(rb_cTensor)) {
      indices.push_back(TensorIndex(from_ruby<Tensor>(obj)));
    } else if (obj.is_nil()) {
      indices.push_back(TensorIndex(torch::indexing::None));
    } else if (obj == True || obj == False) {
      indices.push_back(TensorIndex(from_ruby<bool>(obj)));
    } else {
      throw Exception(rb_eArgError, "Unsupported index type: %s", rb_obj_classname(obj));
    }
  }
  return indices;
}

extern "C"
void Init_ext()
{
  Module rb_mTorch = define_module("Torch");
  rb_mTorch.add_handler<torch::Error>(handle_error);
  add_torch_functions(rb_mTorch);

  rb_cTensor = define_class_under<torch::Tensor>(rb_mTorch, "Tensor");
  rb_cTensor.add_handler<torch::Error>(handle_error);
  add_tensor_functions(rb_cTensor);

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

  // https://pytorch.org/cppdocs/api/structc10_1_1_i_value.html
  Class rb_cIValue = define_class_under<torch::IValue>(rb_mTorch, "IValue")
    .add_handler<torch::Error>(handle_error)
    .define_constructor(Constructor<torch::IValue>())
    .define_method("bool?", &torch::IValue::isBool)
    .define_method("bool_list?", &torch::IValue::isBoolList)
    .define_method("capsule?", &torch::IValue::isCapsule)
    .define_method("custom_class?", &torch::IValue::isCustomClass)
    .define_method("device?", &torch::IValue::isDevice)
    .define_method("double?", &torch::IValue::isDouble)
    .define_method("double_list?", &torch::IValue::isDoubleList)
    .define_method("future?", &torch::IValue::isFuture)
    // .define_method("generator?", &torch::IValue::isGenerator)
    .define_method("generic_dict?", &torch::IValue::isGenericDict)
    .define_method("list?", &torch::IValue::isList)
    .define_method("int?", &torch::IValue::isInt)
    .define_method("int_list?", &torch::IValue::isIntList)
    .define_method("module?", &torch::IValue::isModule)
    .define_method("none?", &torch::IValue::isNone)
    .define_method("object?", &torch::IValue::isObject)
    .define_method("ptr_type?", &torch::IValue::isPtrType)
    .define_method("py_object?", &torch::IValue::isPyObject)
    .define_method("r_ref?", &torch::IValue::isRRef)
    .define_method("scalar?", &torch::IValue::isScalar)
    .define_method("string?", &torch::IValue::isString)
    .define_method("tensor?", &torch::IValue::isTensor)
    .define_method("tensor_list?", &torch::IValue::isTensorList)
    .define_method("tuple?", &torch::IValue::isTuple)
    .define_method(
      "to_bool",
      *[](torch::IValue& self) {
        return self.toBool();
      })
    .define_method(
      "to_double",
      *[](torch::IValue& self) {
        return self.toDouble();
      })
    .define_method(
      "to_int",
      *[](torch::IValue& self) {
        return self.toInt();
      })
    .define_method(
      "to_list",
      *[](torch::IValue& self) {
        auto list = self.toListRef();
        Array obj;
        for (auto& elem : list) {
          obj.push(to_ruby<torch::IValue>(torch::IValue{elem}));
        }
        return obj;
      })
    .define_method(
      "to_string_ref",
      *[](torch::IValue& self) {
        return self.toStringRef();
      })
    .define_method(
      "to_tensor",
      *[](torch::IValue& self) {
        return self.toTensor();
      })
    .define_method(
      "to_generic_dict",
      *[](torch::IValue& self) {
        auto dict = self.toGenericDict();
        Hash obj;
        for (auto& pair : dict) {
          obj[to_ruby<torch::IValue>(torch::IValue{pair.key()})] = to_ruby<torch::IValue>(torch::IValue{pair.value()});
        }
        return obj;
      })
    .define_singleton_method(
      "from_tensor",
      *[](torch::Tensor& v) {
        return torch::IValue(v);
      })
    // TODO create specialized list types?
    .define_singleton_method(
      "from_list",
      *[](Array obj) {
        c10::impl::GenericList list(c10::AnyType::get());
        for (auto entry : obj) {
          list.push_back(from_ruby<torch::IValue>(entry));
        }
        return torch::IValue(list);
      })
    .define_singleton_method(
      "from_string",
      *[](String v) {
        return torch::IValue(v.str());
      })
    .define_singleton_method(
      "from_int",
      *[](int64_t v) {
        return torch::IValue(v);
      })
    .define_singleton_method(
      "from_double",
      *[](double v) {
        return torch::IValue(v);
      })
    .define_singleton_method(
      "from_bool",
      *[](bool v) {
        return torch::IValue(v);
      })
    // see https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/python/pybind_utils.h
    // createGenericDict and toIValue
    .define_singleton_method(
      "from_dict",
      *[](Hash obj) {
        auto key_type = c10::AnyType::get();
        auto value_type = c10::AnyType::get();
        c10::impl::GenericDict elems(key_type, value_type);
        elems.reserve(obj.size());
        for (auto entry : obj) {
          elems.insert(from_ruby<torch::IValue>(entry.first), from_ruby<torch::IValue>((Object) entry.second));
        }
        return torch::IValue(std::move(elems));
      });

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
    // begin tensor creation
    .define_singleton_method(
      "_arange",
      *[](Scalar start, Scalar end, Scalar step, const torch::TensorOptions &options) {
        return torch::arange(start, end, step, options);
      })
    .define_singleton_method(
      "_empty",
      *[](std::vector<int64_t> size, const torch::TensorOptions &options) {
        return torch::empty(size, options);
      })
    .define_singleton_method(
      "_eye",
      *[](int64_t m, int64_t n, const torch::TensorOptions &options) {
        return torch::eye(m, n, options);
      })
    .define_singleton_method(
      "_full",
      *[](std::vector<int64_t> size, Scalar fill_value, const torch::TensorOptions& options) {
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
      *[](std::vector<int64_t> size, const torch::TensorOptions &options) {
        return torch::ones(size, options);
      })
    .define_singleton_method(
      "_rand",
      *[](std::vector<int64_t> size, const torch::TensorOptions &options) {
        return torch::rand(size, options);
      })
    .define_singleton_method(
      "_randint",
      *[](int64_t low, int64_t high, std::vector<int64_t> size, const torch::TensorOptions &options) {
        return torch::randint(low, high, size, options);
      })
    .define_singleton_method(
      "_randn",
      *[](std::vector<int64_t> size, const torch::TensorOptions &options) {
        return torch::randn(size, options);
      })
    .define_singleton_method(
      "_randperm",
      *[](int64_t n, const torch::TensorOptions &options) {
        return torch::randperm(n, options);
      })
    .define_singleton_method(
      "_zeros",
      *[](std::vector<int64_t> size, const torch::TensorOptions &options) {
        return torch::zeros(size, options);
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
      "grad",
      *[](Tensor& self) {
        auto grad = self.grad();
        return grad.defined() ? to_ruby<torch::Tensor>(grad) : Nil;
      })
    .define_method(
      "grad=",
      *[](Tensor& self, torch::Tensor& grad) {
        self.grad() = grad;
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
        self.grad() = grad;
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
    .define_singleton_method("device_count", &torch::cuda::device_count);
}
