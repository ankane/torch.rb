// adapted from PyTorch - python_arg_parser.h

#pragma once

#include <torch/torch.h>
#include <rice/Exception.hpp>
#include "templates.hpp"

extern VALUE THPVariableClass;

enum class ParameterType {
  TENSOR, SCALAR, INT64, DOUBLE, COMPLEX, TENSOR_LIST, INT_LIST, GENERATOR,
  BOOL, STORAGE, PYOBJECT, SCALARTYPE, LAYOUT, MEMORY_FORMAT, DEVICE, STRING,
  DIMNAME, DIMNAME_LIST, QSCHEME, FLOAT_LIST
};

struct FunctionParameter {
  FunctionParameter(const std::string& fmt, bool keyword_only);

  bool check(VALUE obj, int argnum);

  void set_default_str(const std::string& str);
  std::string type_name() const;

  ParameterType type_;
  bool optional;
  bool allow_none;
  bool keyword_only;
  bool allow_numbers_as_tensors = false;
  int size;
  std::string name;
  VALUE ruby_name;
  at::SmallVector<VALUE, 5> numpy_python_names;
  at::Scalar default_scalar;
  std::vector<int64_t> default_intlist;
  union {
    bool default_bool;
    int64_t default_int;
    double default_double;
    double default_complex[2]; // see Scalar
    at::ScalarType default_scalartype;
    at::Layout default_layout;
  };
};

struct FunctionSignature {
  explicit FunctionSignature(const std::string& fmt, int index);

  bool parse(VALUE self, VALUE args, VALUE kwargs, std::vector<VALUE>& dst, bool raise_exception);

  std::string toString() const;

  std::string name;
  std::vector<FunctionParameter> params;
  // std::vector<py::handle> overloaded_args;
  ssize_t min_args;
  ssize_t max_args;
  ssize_t max_pos_args;
  int index;
  bool hidden;
  bool deprecated;
  bool disable_torch_function;
};

// keep THP prefix for now to make it easier to compare code

inline VALUE THPUtils_internSymbol(const std::string& str) {
  return Symbol(str);
}

inline std::string THPUtils_unpackSymbol(VALUE obj) {
  Check_Type(obj, T_SYMBOL);
  obj = rb_funcall(obj, rb_intern("to_s"), 0);
  return std::string(RSTRING_PTR(obj), RSTRING_LEN(obj));
}

inline std::string THPUtils_unpackString(VALUE obj) {
  Check_Type(obj, T_STRING);
  return std::string(RSTRING_PTR(obj), RSTRING_LEN(obj));
}

inline bool THPUtils_checkSymbol(VALUE obj) {
  return SYMBOL_P(obj);
}

inline bool THPUtils_checkIndex(VALUE obj) {
  return FIXNUM_P(obj);
}

inline bool THPUtils_checkScalar(VALUE obj) {
  return FIXNUM_P(obj) || RB_FLOAT_TYPE_P(obj) || RB_TYPE_P(obj, T_COMPLEX);
}

inline bool THPVariable_Check(VALUE obj) {
  return rb_obj_is_kind_of(obj, THPVariableClass);
}

inline bool THPVariable_CheckExact(VALUE obj) {
  return rb_obj_is_instance_of(obj, THPVariableClass);
}

struct RubyArgs {
  RubyArgs(const FunctionSignature& signature, std::vector<VALUE> &args)
    : signature(signature)
    , args(args)
    , idx(signature.index) {}

  const FunctionSignature& signature;
  std::vector<VALUE> args;
  int idx;

  torch::Tensor tensor(int i) {
    return from_ruby<torch::Tensor>(args[i]);
  }

  std::vector<int64_t> intlist(int i) {
    if (NIL_P(args[i])) return signature.params[i].default_intlist;

    VALUE arg = args[i];
    auto size = signature.params[i].size;
    if (size > 0 && FIXNUM_P(arg)) {
      return std::vector<int64_t>(size, FIX2INT(arg));
    }

    size = RARRAY_LEN(arg);
    std::vector<int64_t> res(size);
    for (idx = 0; idx < size; idx++) {
      VALUE obj = rb_ary_entry(arg, idx);
      if (FIXNUM_P(obj)) {
        res[idx] = from_ruby<int64_t>(obj);
      } else {
        rb_raise(rb_eArgError, "%s(): argument '%s' must be %s, but found element of type %s at pos %d",
            signature.name.c_str(), signature.params[i].name.c_str(),
            signature.params[i].type_name().c_str(), rb_obj_classname(obj), idx + 1);
      }
    }
    return res;
  }

  Scalar scalar(int i) {
    if (NIL_P(args[i])) return signature.params[i].default_scalar;
    return from_ruby<torch::Scalar>(args[i]);
  }

  c10::optional<Scalar> scalarOptional(int i) {
    if (NIL_P(args[i])) return c10::nullopt;
    return scalar(i);
  }

  bool toBool(int i) {
    if (NIL_P(args[i])) return signature.params[i].default_bool;
    return RTEST(args[i]);
  }

  int64_t toInt64(int i) {
    if (NIL_P(args[i])) return signature.params[i].default_int;
    return from_ruby<int64_t>(args[i]);
  }

  double toDouble(int i) {
    if (NIL_P(args[i])) return signature.params[i].default_double;
    return from_ruby<double>(args[i]);
  }

  c10::optional<double> toDoubleOptional(int i) {
    if (NIL_P(args[i])) return c10::nullopt;
    return toDouble(i);
  }

  c10::optional<int64_t> toInt64Optional(int i) {
    if (NIL_P(args[i])) return c10::nullopt;
    return toInt64(i);
  }

  ScalarType scalartype(int i) {
    if (NIL_P(args[i])) {
      auto scalartype = signature.params[i].default_scalartype;
      return (scalartype == at::ScalarType::Undefined) ? at::typeMetaToScalarType(at::get_default_dtype()) : scalartype;
    }

    static std::unordered_map<VALUE, ScalarType> dtype_map = {
      {ID2SYM(rb_intern("uint8")), ScalarType::Byte},
      {ID2SYM(rb_intern("int8")), ScalarType::Char},
      {ID2SYM(rb_intern("short")), ScalarType::Short},
      {ID2SYM(rb_intern("int16")), ScalarType::Short},
      {ID2SYM(rb_intern("int")), ScalarType::Int},
      {ID2SYM(rb_intern("int32")), ScalarType::Int},
      {ID2SYM(rb_intern("long")), ScalarType::Long},
      {ID2SYM(rb_intern("int64")), ScalarType::Long},
      {ID2SYM(rb_intern("float")), ScalarType::Float},
      {ID2SYM(rb_intern("float32")), ScalarType::Float},
      {ID2SYM(rb_intern("double")), ScalarType::Double},
      {ID2SYM(rb_intern("float64")), ScalarType::Double},
      {ID2SYM(rb_intern("complex_half")), ScalarType::ComplexHalf},
      {ID2SYM(rb_intern("complex_float")), ScalarType::ComplexFloat},
      {ID2SYM(rb_intern("complex_double")), ScalarType::ComplexDouble},
      {ID2SYM(rb_intern("bool")), ScalarType::Bool},
      {ID2SYM(rb_intern("qint8")), ScalarType::QInt8},
      {ID2SYM(rb_intern("quint8")), ScalarType::QUInt8},
      {ID2SYM(rb_intern("qint32")), ScalarType::QInt32},
      {ID2SYM(rb_intern("bfloat16")), ScalarType::BFloat16},
    };

    auto it = dtype_map.find(args[i]);
    if (it == dtype_map.end()) {
      rb_raise(rb_eArgError, "invalid dtype: %s", THPUtils_unpackSymbol(args[i]).c_str());
    }
    return it->second;
  }

  c10::optional<ScalarType> scalartypeOptional(int i) {
    if (NIL_P(args[i])) return c10::nullopt;
    return scalartype(i);
  }

  std::string toString(int i) {
    return from_ruby<std::string>(args[i]);
  }

  c10::optional<bool> toBoolOptional(int i) {
    if (NIL_P(args[i])) return c10::nullopt;
    return toBool(i);
  }

  OptionalTensor optionalTensor(int i) {
    if (NIL_P(args[i])) return OptionalTensor(Nil);
    return tensor(i);
  }

  std::vector<at::Tensor> tensorlist(int i) {
    if (NIL_P(args[i])) return std::vector<at::Tensor>();
    return from_ruby<std::vector<Tensor>>(args[i]);
  }

  c10::optional<at::Generator> generator(int i) {
    if (NIL_P(args[i])) return c10::nullopt;
    throw std::runtime_error("generator not supported yet");
  }

  at::MemoryFormat memoryformat(int i) {
    if (NIL_P(args[i])) return at::MemoryFormat::Contiguous;
    throw std::runtime_error("memoryformat not supported yet");
  }

  c10::optional<at::MemoryFormat> memoryformatOptional(int i) {
    if (NIL_P(args[i])) return c10::nullopt;
    return memoryformat(i);
  }

  at::Storage storage(int i) {
    if (NIL_P(args[i])) return at::Storage();
    throw std::runtime_error("storage not supported yet");
  }

  c10::optional<at::Layout> layoutOptional(int i) {
    if (NIL_P(args[i])) return c10::nullopt;

    static std::unordered_map<VALUE, Layout> layout_map = {
      {ID2SYM(rb_intern("strided")), Layout::Strided},
    };

    auto it = layout_map.find(args[i]);
    if (it == layout_map.end()) {
      rb_raise(rb_eArgError, "invalid layout: %s", THPUtils_unpackSymbol(args[i]).c_str());
    }
    return it->second;
  }

  at::Device device(int i) {
    if (NIL_P(args[i])) {
      return at::Device("cpu");
    }
    const std::string &device_str = THPUtils_unpackString(args[i]);
    return at::Device(device_str);
  }

  bool isNone(int i) {
    return NIL_P(args[i]);
  }

  template<int N>
  std::array<at::Tensor, N> tensorlist_n(int i) {
    auto res = std::array<at::Tensor, N>();
    if (NIL_P(args[i])) return res;
    VALUE arg = args[i];
    Check_Type(arg, T_ARRAY);
    auto size = RARRAY_LEN(arg);
    if (size != N) {
      rb_raise(rb_eArgError, "expected array of %d elements but got %d", N, (int)size);
    }
    for (int idx = 0; idx < size; idx++) {
      VALUE obj = rb_ary_entry(arg, idx);
      res[idx] = from_ruby<Tensor>(obj);
    }
    return res;
  }
};

struct RubyArgParser {
  std::vector<FunctionSignature> signatures_;
  std::string function_name;
  ssize_t max_args;

  public:
    RubyArgParser(std::vector<std::string> fmts) : max_args(0) {
      int index = 0;
      for (auto& fmt : fmts) {
        signatures_.emplace_back(fmt, index);
        ++index;
      }
      for (auto& signature : signatures_) {
        if (signature.max_args > max_args) {
          max_args = signature.max_args;
        }
      }
      if (signatures_.size() > 0) {
        function_name = signatures_[0].name;
      }

      // Check deprecated signatures last
      std::stable_partition(signatures_.begin(), signatures_.end(),
        [](const FunctionSignature & sig) {
          return !sig.deprecated;
        });
    }

    RubyArgs parse(VALUE self, int argc, VALUE* argv, std::vector<VALUE> &parsed_args) {
      VALUE args, kwargs;
      rb_scan_args(argc, argv, "*:", &args, &kwargs);

      if (signatures_.size() == 1) {
        auto& signature = signatures_[0];
        signature.parse(self, args, kwargs, parsed_args, true);
        return RubyArgs(signature, parsed_args);
      }

      for (auto& signature : signatures_) {
        if (signature.parse(self, args, kwargs, parsed_args, false)) {
          return RubyArgs(signature, parsed_args);
        }
      }

      print_error(self, args, kwargs, parsed_args);

      // TODO better message
      rb_raise(rb_eArgError, "No matching signatures");
    }

    void print_error(VALUE self, VALUE args, VALUE kwargs, std::vector<VALUE>& parsed_args) {
      ssize_t num_args = (NIL_P(args) ? 0 : RARRAY_LEN(args)) + (NIL_P(kwargs) ? 0 : RHASH_SIZE(kwargs));
      std::vector<int> plausible_idxs;
      ssize_t i = 0;
      for (auto& signature : signatures_) {
        if (num_args >= signature.min_args && num_args <= signature.max_args && !signature.hidden) {
          plausible_idxs.push_back(i);
        }
        i++;
      }

      if (plausible_idxs.size() == 1) {
        auto& signature = signatures_[plausible_idxs[0]];
        signature.parse(self, args, kwargs, parsed_args, true);
      }
    }
};
