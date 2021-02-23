#include <torch/torch.h>

#include <rice/Array.hpp>
#include <rice/Class.hpp>
#include <rice/Hash.hpp>
#include <rice/String.hpp>

void init_ivalue(Rice::Class& c) {
  c
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
        Rice::Array obj;
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
        Rice::Hash obj;
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
      *[](Rice::Array obj) {
        c10::impl::GenericList list(c10::AnyType::get());
        for (auto entry : obj) {
          list.push_back(from_ruby<torch::IValue>(entry));
        }
        return torch::IValue(list);
      })
    .define_singleton_method(
      "from_string",
      *[](Rice::String v) {
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
      *[](Rice::Hash obj) {
        auto key_type = c10::AnyType::get();
        auto value_type = c10::AnyType::get();
        c10::impl::GenericDict elems(key_type, value_type);
        elems.reserve(obj.size());
        for (auto entry : obj) {
          elems.insert(from_ruby<torch::IValue>(entry.first), from_ruby<torch::IValue>((Rice::Object) entry.second));
        }
        return torch::IValue(std::move(elems));
      });
}
