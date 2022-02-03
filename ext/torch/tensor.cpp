#include <torch/torch.h>

#include <rice/rice.hpp>

#include "tensor_functions.h"
#include "ruby_arg_parser.h"
#include "templates.h"
#include "utils.h"

using namespace Rice;
using torch::indexing::TensorIndex;

template<typename T>
Array flat_data(Tensor& tensor) {
  Tensor view = tensor.reshape({tensor.numel()});

  Array a;
  for (int i = 0; i < tensor.numel(); i++) {
    a.push(view[i].item().to<T>());
  }
  return a;
}

Class rb_cTensor;

std::vector<TensorIndex> index_vector(Array a) {
  Object obj;

  std::vector<TensorIndex> indices;
  indices.reserve(a.size());

  for (long i = 0; i < a.size(); i++) {
    obj = a[i];

    if (obj.is_instance_of(rb_cInteger)) {
      indices.push_back(Rice::detail::From_Ruby<int64_t>().convert(obj.value()));
    } else if (obj.is_instance_of(rb_cRange)) {
      torch::optional<int64_t> start_index = torch::nullopt;
      torch::optional<int64_t> stop_index = torch::nullopt;

      Object begin = obj.call("begin");
      if (!begin.is_nil()) {
        start_index = Rice::detail::From_Ruby<int64_t>().convert(begin.value());
      }

      Object end = obj.call("end");
      if (!end.is_nil()) {
        stop_index = Rice::detail::From_Ruby<int64_t>().convert(end.value());
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
      indices.push_back(Rice::detail::From_Ruby<Tensor>().convert(obj.value()));
    } else if (obj.is_nil()) {
      indices.push_back(torch::indexing::None);
    } else if (obj == True || obj == False) {
      indices.push_back(Rice::detail::From_Ruby<bool>().convert(obj.value()));
    } else {
      throw Exception(rb_eArgError, "Unsupported index type: %s", rb_obj_classname(obj));
    }
  }
  return indices;
}

// hack (removes inputs argument)
// https://github.com/pytorch/pytorch/commit/2e5bfa9824f549be69a28e4705a72b4cf8a4c519
// TODO add support for inputs argument
// _backward
static VALUE tensor__backward(int argc, VALUE* argv, VALUE self_)
{
  HANDLE_TH_ERRORS
  Tensor& self = Rice::detail::From_Ruby<Tensor&>().convert(self_);
  static RubyArgParser parser({
    "_backward(Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False)"
  });
  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(self_, argc, argv, parsed_args);
  // _backward(Tensor self, Tensor[] inputs, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()
  auto dispatch__backward = [](const Tensor & self, TensorList inputs, const c10::optional<at::Tensor> & gradient, c10::optional<bool> retain_graph, bool create_graph) -> void {
    // in future, release GVL
    self._backward(inputs, gradient, retain_graph, create_graph);
  };
  dispatch__backward(self, {}, _r.optionalTensor(0), _r.toBoolOptional(1), _r.toBool(2));
  RETURN_NIL
  END_HANDLE_TH_ERRORS
}

void init_tensor(Rice::Module& m, Rice::Class& c, Rice::Class& rb_cTensorOptions) {
  rb_cTensor = c;
  rb_cTensor.add_handler<torch::Error>(handle_error);
  add_tensor_functions(rb_cTensor);
  THPVariableClass = rb_cTensor.value();

  rb_define_method(rb_cTensor, "backward", (VALUE (*)(...)) tensor__backward, -1);

  rb_cTensor
    .define_method("cuda?", [](Tensor& self) { return self.is_cuda(); })
    .define_method("sparse?", [](Tensor& self) { return self.is_sparse(); })
    .define_method("quantized?", [](Tensor& self) { return self.is_quantized(); })
    .define_method("dim", [](Tensor& self) { return self.dim(); })
    .define_method("numel", [](Tensor& self) { return self.numel(); })
    .define_method("element_size", [](Tensor& self) { return self.element_size(); })
    .define_method("requires_grad", [](Tensor& self) { return self.requires_grad(); })
    .define_method(
      "_size",
      [](Tensor& self, int64_t dim) {
        return self.size(dim);
      })
    .define_method(
      "_stride",
      [](Tensor& self, int64_t dim) {
        return self.stride(dim);
      })
    // in C++ for performance
    .define_method(
      "shape",
      [](Tensor& self) {
        Array a;
        for (auto &size : self.sizes()) {
          a.push(size);
        }
        return a;
      })
    .define_method(
      "_strides",
      [](Tensor& self) {
        Array a;
        for (auto &stride : self.strides()) {
          a.push(stride);
        }
        return a;
      })
    .define_method(
      "_index",
      [](Tensor& self, Array indices) {
        auto vec = index_vector(indices);
        return self.index(vec);
      })
    .define_method(
      "_index_put_custom",
      [](Tensor& self, Array indices, torch::Tensor& value) {
        auto vec = index_vector(indices);
        return self.index_put_(vec, value);
      })
    .define_method(
      "contiguous?",
      [](Tensor& self) {
        return self.is_contiguous();
      })
    .define_method(
      "_requires_grad!",
      [](Tensor& self, bool requires_grad) {
        return self.set_requires_grad(requires_grad);
      })
    .define_method(
      "grad",
      [](Tensor& self) {
        auto grad = self.grad();
        return grad.defined() ? Object(Rice::detail::To_Ruby<torch::Tensor>().convert(grad)) : Nil;
      })
    .define_method(
      "grad=",
      [](Tensor& self, Rice::Object value) {
        if (value.is_nil()) {
          self.mutable_grad().reset();
          return value;
        }

        const auto& grad = Rice::detail::From_Ruby<torch::Tensor>().convert(value.value());

        // TODO support sparse grad
        if (!grad.options().type_equal(self.options())) {
          rb_raise(rb_eArgError, "assigned grad has data of a different type");
        }

        if (self.is_cuda() && grad.get_device() != self.get_device()) {
          rb_raise(rb_eArgError, "assigned grad has data located on a different device");
        }

        if (!self.sizes().equals(grad.sizes())) {
          rb_raise(rb_eArgError, "assigned grad has data of a different size");
        }

        self.mutable_grad() = grad;

        return value;
      })
    .define_method(
      "_dtype",
      [](Tensor& self) {
        return (int) at::typeMetaToScalarType(self.dtype());
      })
    .define_method(
      "_type",
      [](Tensor& self, int dtype) {
        return self.toType((torch::ScalarType) dtype);
      })
    .define_method(
      "_layout",
      [](Tensor& self) {
        std::stringstream s;
        s << self.layout();
        return s.str();
      })
    .define_method(
      "device",
      [](Tensor& self) {
        std::stringstream s;
        s << self.device();
        return s.str();
      })
    .define_method(
      "_data_str",
      [](Tensor& self) {
        auto tensor = self;

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
      [](Tensor& self) {
        return reinterpret_cast<uintptr_t>(self.data_ptr());
      })
    // TODO figure out a better way to do this
    .define_method(
      "_flat_data",
      [](Tensor& self) {
        auto tensor = self;

        // move to CPU to get data
        if (tensor.device().type() != torch::kCPU) {
          torch::Device device("cpu");
          tensor = tensor.to(device);
        }

        auto dtype = tensor.dtype();
        if (dtype == torch::kByte) {
          return flat_data<uint8_t>(tensor);
        } else if (dtype == torch::kChar) {
          return flat_data<int8_t>(tensor);
        } else if (dtype == torch::kShort) {
          return flat_data<int16_t>(tensor);
        } else if (dtype == torch::kInt) {
          return flat_data<int32_t>(tensor);
        } else if (dtype == torch::kLong) {
          return flat_data<int64_t>(tensor);
        } else if (dtype == torch::kFloat) {
          return flat_data<float>(tensor);
        } else if (dtype == torch::kDouble) {
          return flat_data<double>(tensor);
        } else if (dtype == torch::kBool) {
          return flat_data<bool>(tensor);
        } else if (dtype == torch::kComplexFloat) {
          return flat_data<c10::complex<float>>(tensor);
        } else if (dtype == torch::kComplexDouble) {
          return flat_data<c10::complex<double>>(tensor);
        } else {
          throw std::runtime_error("Unsupported type");
        }
      })
    .define_method(
      "_to",
      [](Tensor& self, torch::Device& device, int dtype, bool non_blocking, bool copy) {
        return self.to(device, (torch::ScalarType) dtype, non_blocking, copy);
      });

  rb_cTensorOptions
    .add_handler<torch::Error>(handle_error)
    .define_method(
      "dtype",
      [](torch::TensorOptions& self, int dtype) {
        return self.dtype((torch::ScalarType) dtype);
      })
    .define_method(
      "layout",
      [](torch::TensorOptions& self, const std::string& layout) {
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
      [](torch::TensorOptions& self, const std::string& device) {
        torch::Device d(device);
        return self.device(d);
      })
    .define_method(
      "requires_grad",
      [](torch::TensorOptions& self, bool requires_grad) {
        return self.requires_grad(requires_grad);
      });
}
