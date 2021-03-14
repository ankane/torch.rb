// adapted from PyTorch - python_arg_parser.h

#pragma once

#include <sstream>

#include <torch/torch.h>
#include <rice/Exception.hpp>

#include "templates.h"
#include "utils.h"

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

  bool parse(VALUE self, VALUE args, VALUE kwargs, VALUE dst[], bool raise_exception);

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

struct RubyArgs {
  RubyArgs(const FunctionSignature& signature, VALUE* args)
    : signature(signature)
    , args(args)
    , idx(signature.index) {}

  const FunctionSignature& signature;
  VALUE* args;
  int idx;

  inline at::Tensor tensor(int i);
  inline OptionalTensor optionalTensor(int i);
  inline at::Scalar scalar(int i);
  // inline at::Scalar scalarWithDefault(int i, at::Scalar default_scalar);
  inline std::vector<at::Tensor> tensorlist(int i);
  template<int N>
  inline std::array<at::Tensor, N> tensorlist_n(int i);
  inline std::vector<int64_t> intlist(int i);
  // inline c10::OptionalArray<int64_t> intlistOptional(int i);
  // inline std::vector<int64_t> intlistWithDefault(int i, std::vector<int64_t> default_intlist);
  inline c10::optional<at::Generator> generator(int i);
  inline at::Storage storage(int i);
  inline at::ScalarType scalartype(int i);
  // inline at::ScalarType scalartypeWithDefault(int i, at::ScalarType default_scalartype);
  inline c10::optional<at::ScalarType> scalartypeOptional(int i);
  inline c10::optional<at::Scalar> scalarOptional(int i);
  inline c10::optional<int64_t> toInt64Optional(int i);
  inline c10::optional<bool> toBoolOptional(int i);
  inline c10::optional<double> toDoubleOptional(int i);
  inline c10::OptionalArray<double> doublelistOptional(int i);
  // inline at::Layout layout(int i);
  // inline at::Layout layoutWithDefault(int i, at::Layout default_layout);
  inline c10::optional<at::Layout> layoutOptional(int i);
  inline at::Device device(int i);
  // inline at::Device deviceWithDefault(int i, const at::Device& default_device);
  // inline c10::optional<at::Device> deviceOptional(int i);
  // inline at::Dimname dimname(int i);
  // inline std::vector<at::Dimname> dimnamelist(int i);
  // inline c10::optional<std::vector<at::Dimname>> toDimnameListOptional(int i);
  inline at::MemoryFormat memoryformat(int i);
  inline c10::optional<at::MemoryFormat> memoryformatOptional(int i);
  // inline at::QScheme toQScheme(int i);
  inline std::string string(int i);
  inline c10::optional<std::string> stringOptional(int i);
  // inline PyObject* pyobject(int i);
  inline int64_t toInt64(int i);
  // inline int64_t toInt64WithDefault(int i, int64_t default_int);
  inline double toDouble(int i);
  // inline double toDoubleWithDefault(int i, double default_double);
  // inline c10::complex<double> toComplex(int i);
  // inline c10::complex<double> toComplexWithDefault(int i, c10::complex<double> default_complex);
  inline bool toBool(int i);
  // inline bool toBoolWithDefault(int i, bool default_bool);
  inline bool isNone(int i);
};

inline at::Tensor RubyArgs::tensor(int i) {
  return from_ruby<torch::Tensor>(args[i]);
}

inline OptionalTensor RubyArgs::optionalTensor(int i) {
  if (NIL_P(args[i])) return OptionalTensor(Nil);
  return tensor(i);
}

inline at::Scalar RubyArgs::scalar(int i) {
  if (NIL_P(args[i])) return signature.params[i].default_scalar;
  return from_ruby<torch::Scalar>(args[i]);
}

inline std::vector<at::Tensor> RubyArgs::tensorlist(int i) {
  if (NIL_P(args[i])) return std::vector<at::Tensor>();
  return from_ruby<std::vector<Tensor>>(args[i]);
}

template<int N>
inline std::array<at::Tensor, N> RubyArgs::tensorlist_n(int i) {
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

inline std::vector<int64_t> RubyArgs::intlist(int i) {
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

inline c10::optional<at::Generator> RubyArgs::generator(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  throw std::runtime_error("generator not supported yet");
}

inline at::Storage RubyArgs::storage(int i) {
  if (NIL_P(args[i])) return at::Storage();
  throw std::runtime_error("storage not supported yet");
}

inline ScalarType RubyArgs::scalartype(int i) {
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

inline c10::optional<ScalarType> RubyArgs::scalartypeOptional(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  return scalartype(i);
}

inline c10::optional<Scalar> RubyArgs::scalarOptional(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  return scalar(i);
}

inline c10::optional<int64_t> RubyArgs::toInt64Optional(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  return toInt64(i);
}

inline c10::optional<bool> RubyArgs::toBoolOptional(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  return toBool(i);
}

inline c10::optional<double> RubyArgs::toDoubleOptional(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  return toDouble(i);
}

inline c10::OptionalArray<double> RubyArgs::doublelistOptional(int i) {
  if (NIL_P(args[i])) return {};

  VALUE arg = args[i];
  auto size = RARRAY_LEN(arg);
  std::vector<double> res(size);
  for (idx = 0; idx < size; idx++) {
    VALUE obj = rb_ary_entry(arg, idx);
    if (FIXNUM_P(obj) || RB_FLOAT_TYPE_P(obj)) {
      res[idx] = from_ruby<double>(obj);
    } else {
      rb_raise(rb_eArgError, "%s(): argument '%s' must be %s, but found element of type %s at pos %d",
          signature.name.c_str(), signature.params[i].name.c_str(),
          signature.params[i].type_name().c_str(), rb_obj_classname(obj), idx + 1);
    }
  }
  return res;
}

inline c10::optional<at::Layout> RubyArgs::layoutOptional(int i) {
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

inline at::Device RubyArgs::device(int i) {
  if (NIL_P(args[i])) {
    return at::Device("cpu");
  }
  const std::string &device_str = THPUtils_unpackString(args[i]);
  return at::Device(device_str);
}

inline at::MemoryFormat RubyArgs::memoryformat(int i) {
  if (NIL_P(args[i])) return at::MemoryFormat::Contiguous;
  throw std::runtime_error("memoryformat not supported yet");
}

inline c10::optional<at::MemoryFormat> RubyArgs::memoryformatOptional(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  return memoryformat(i);
}

inline std::string RubyArgs::string(int i) {
  return from_ruby<std::string>(args[i]);
}

inline c10::optional<std::string> RubyArgs::stringOptional(int i) {
  if (!args[i]) return c10::nullopt;
  return from_ruby<std::string>(args[i]);
}

inline int64_t RubyArgs::toInt64(int i) {
  if (NIL_P(args[i])) return signature.params[i].default_int;
  return from_ruby<int64_t>(args[i]);
}

inline double RubyArgs::toDouble(int i) {
  if (NIL_P(args[i])) return signature.params[i].default_double;
  return from_ruby<double>(args[i]);
}

inline bool RubyArgs::toBool(int i) {
  if (NIL_P(args[i])) return signature.params[i].default_bool;
  return RTEST(args[i]);
}

inline bool RubyArgs::isNone(int i) {
  return NIL_P(args[i]);
}

template<int N>
struct ParsedArgs {
  ParsedArgs() : args() { }
  VALUE args[N];
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

    template<int N>
    inline RubyArgs parse(VALUE self, int argc, VALUE* argv, ParsedArgs<N> &dst) {
      if (N < max_args) {
        rb_raise(rb_eArgError, "RubyArgParser: dst ParsedArgs buffer does not have enough capacity, expected %d (got %d)", (int)max_args, N);
      }
      return raw_parse(self, argc, argv, dst.args);
    }

    inline RubyArgs raw_parse(VALUE self, int argc, VALUE* argv, VALUE parsed_args[]) {
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

    void print_error(VALUE self, VALUE args, VALUE kwargs, VALUE parsed_args[]) {
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
