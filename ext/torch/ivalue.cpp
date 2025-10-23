#include <utility>

#include <torch/torch.h>

#include <rice/rice.hpp>

#include "utils.h"

void init_ivalue(Rice::Module& m, Rice::Class& rb_cIValue) {
  // https://pytorch.org/cppdocs/api/structc10_1_1_i_value.html
  rb_cIValue
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
      [](torch::IValue& self) {
        return self.toBool();
      })
    .define_method(
      "to_double",
      [](torch::IValue& self) {
        return self.toDouble();
      })
    .define_method(
      "to_int",
      [](torch::IValue& self) {
        return self.toInt();
      })
    .define_method(
      "to_list",
      [](torch::IValue& self) {
        auto list = self.toListRef();
        Rice::Array obj;
        for (auto& elem : list) {
          auto v = torch::IValue{elem};
          obj.push(Rice::Object(Rice::detail::To_Ruby<torch::IValue>().convert(v)), false);
        }
        return obj;
      })
    .define_method(
      "to_string_ref",
      [](torch::IValue& self) {
        return self.toStringRef();
      })
    .define_method(
      "to_tensor",
      [](torch::IValue& self) {
        return self.toTensor();
      })
    .define_method(
      "to_generic_dict",
      [](torch::IValue& self) {
        auto dict = self.toGenericDict();
        Rice::Hash obj;
        for (auto& pair : dict) {
          auto k = torch::IValue{pair.key()};
          auto v = torch::IValue{pair.value()};
          obj[Rice::Object(Rice::detail::To_Ruby<torch::IValue>().convert(k))] = Rice::Object(Rice::detail::To_Ruby<torch::IValue>().convert(v));
        }
        return obj;
      })
    .define_singleton_function(
      "from_tensor",
      [](torch::Tensor& v) {
        return torch::IValue(v);
      })
    // TODO create specialized list types?
    .define_singleton_function(
      "from_list",
      [](Rice::Array obj) {
        c10::impl::GenericList list(c10::AnyType::get());
        for (auto entry : obj) {
          list.push_back(Rice::detail::From_Ruby<torch::IValue>().convert(entry.value()));
        }
        return torch::IValue(list);
      })
    .define_singleton_function(
      "from_string",
      [](Rice::String v) {
        return torch::IValue(v.str());
      })
    .define_singleton_function(
      "from_int",
      [](int64_t v) {
        return torch::IValue(v);
      })
    .define_singleton_function(
      "from_double",
      [](double v) {
        return torch::IValue(v);
      })
    .define_singleton_function(
      "from_bool",
      [](bool v) {
        return torch::IValue(v);
      })
    // see https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/python/pybind_utils.h
    // createGenericDict and toIValue
    .define_singleton_function(
      "from_dict",
      [](Rice::Hash obj) {
        auto key_type = c10::AnyType::get();
        auto value_type = c10::AnyType::get();
        c10::impl::GenericDict elems(key_type, value_type);
        elems.reserve(obj.size());
        for (auto entry : obj) {
          elems.insert(Rice::detail::From_Ruby<torch::IValue>().convert(entry.first), Rice::detail::From_Ruby<torch::IValue>().convert((Rice::Object) entry.second));
        }
        return torch::IValue(std::move(elems));
      });
}
