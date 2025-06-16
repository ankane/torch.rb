// adapted from PyTorch - python_arg_parser.h

#pragma once

#include <sstream>
#include <unordered_map>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <rice/rice.hpp>

#include "templates.h"
#include "utils.h"

enum class ParameterType {
  TENSOR, SCALAR, INT64, SYM_INT, DOUBLE, COMPLEX, TENSOR_LIST, INT_LIST, GENERATOR,
  BOOL, STORAGE, PYOBJECT, SCALARTYPE, LAYOUT, MEMORY_FORMAT, DEVICE, STREAM, STRING,
  DIMNAME, DIMNAME_LIST, QSCHEME, FLOAT_LIST, SCALAR_LIST, SYM_INT_LIST
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
  std::string default_string;
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
  inline c10::optional<at::Tensor> optionalTensor(int i);
  inline at::Scalar scalar(int i);
  // inline at::Scalar scalarWithDefault(int i, at::Scalar default_scalar);
  inline std::vector<at::Scalar> scalarlist(int i);
  inline std::vector<at::Tensor> tensorlist(int i);
  template<int N>
  inline std::array<at::Tensor, N> tensorlist_n(int i);
  inline std::vector<int64_t> intlist(int i);
  inline std::vector<c10::SymInt> symintlist(int i);
  inline c10::OptionalArray<int64_t> intlistOptional(int i);
  inline c10::OptionalArray<c10::SymInt> symintlistOptional(int i);
  inline std::vector<int64_t> intlistWithDefault(int i, std::vector<int64_t> default_intlist);
  inline c10::optional<at::Generator> generator(int i);
  inline at::Storage storage(int i);
  inline at::ScalarType scalartype(int i);
  inline at::ScalarType scalartypeWithDefault(int i, at::ScalarType default_scalartype);
  inline c10::optional<at::ScalarType> scalartypeOptional(int i);
  inline c10::optional<at::Scalar> scalarOptional(int i);
  inline c10::optional<int64_t> toInt64Optional(int i);
  inline c10::optional<c10::SymInt> toSymIntOptional(int i);
  inline c10::optional<bool> toBoolOptional(int i);
  inline c10::optional<double> toDoubleOptional(int i);
  inline c10::OptionalArray<double> doublelistOptional(int i);
  inline at::Layout layout(int i);
  inline at::Layout layoutWithDefault(int i, at::Layout default_layout);
  inline c10::optional<at::Layout> layoutOptional(int i);
  inline at::Device device(int i);
  inline at::Device deviceWithDefault(int i, const at::Device& default_device);
  // inline c10::optional<at::Device> deviceOptional(int i);
  // inline at::Dimname dimname(int i);
  // inline std::vector<at::Dimname> dimnamelist(int i);
  // inline c10::optional<std::vector<at::Dimname>> toDimnameListOptional(int i);
  inline at::MemoryFormat memoryformat(int i);
  inline c10::optional<at::MemoryFormat> memoryformatOptional(int i);
  // inline at::QScheme toQScheme(int i);
  inline std::string string(int i);
  inline std::string stringWithDefault(int i, const std::string& default_str);
  inline c10::optional<std::string> stringOptional(int i);
  inline c10::string_view stringView(int i);
  // inline c10::string_view stringViewWithDefault(int i, const c10::string_view default_str);
  inline c10::optional<c10::string_view> stringViewOptional(int i);
  // inline PyObject* pyobject(int i);
  inline int64_t toInt64(int i);
  inline c10::SymInt toSymInt(int i);
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
  return Rice::detail::From_Ruby<torch::Tensor>().convert(args[i]);
}

inline c10::optional<at::Tensor> RubyArgs::optionalTensor(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  return tensor(i);
}

inline at::Scalar RubyArgs::scalar(int i) {
  if (NIL_P(args[i])) return signature.params[i].default_scalar;
  return Rice::detail::From_Ruby<torch::Scalar>().convert(args[i]);
}

inline std::vector<at::Scalar> RubyArgs::scalarlist(int i) {
  if (NIL_P(args[i])) return std::vector<at::Scalar>();
  return Rice::detail::From_Ruby<std::vector<at::Scalar>>().convert(args[i]);
}

inline std::vector<at::Tensor> RubyArgs::tensorlist(int i) {
  if (NIL_P(args[i])) return std::vector<at::Tensor>();
  return Rice::detail::From_Ruby<std::vector<Tensor>>().convert(args[i]);
}

template<int N>
inline std::array<at::Tensor, N> RubyArgs::tensorlist_n(int i) {
  auto res = std::array<at::Tensor, N>();
  if (NIL_P(args[i])) return res;
  VALUE arg = args[i];
  Check_Type(arg, T_ARRAY);
  auto size = RARRAY_LEN(arg);
  if (size != N) {
    rb_raise(rb_eArgError, "expected array of %d elements but got %d", N, static_cast<int>(size));
  }
  for (int idx = 0; idx < size; idx++) {
    VALUE obj = rb_ary_entry(arg, idx);
    res[idx] = Rice::detail::From_Ruby<Tensor>().convert(obj);
  }
  return res;
}

inline std::vector<int64_t> RubyArgs::intlist(int i) {
  return intlistWithDefault(i, signature.params[i].default_intlist);
}

inline std::vector<c10::SymInt> RubyArgs::symintlist(int i) {
  if (NIL_P(args[i])) {
    return c10::fmap(signature.params[i].default_intlist, [](int64_t di) {
      return c10::SymInt(di);
    });
  }

  // TODO improve
  return c10::fmap(intlist(i), [](int64_t di) {
    return c10::SymInt(di);
  });
}

inline std::vector<int64_t> RubyArgs::intlistWithDefault(int i, std::vector<int64_t> default_intlist) {
  if (NIL_P(args[i])) return default_intlist;
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
      res[idx] = Rice::detail::From_Ruby<int64_t>().convert(obj);
    } else {
      rb_raise(rb_eArgError, "%s(): argument '%s' must be %s, but found element of type %s at pos %d",
          signature.name.c_str(), signature.params[i].name.c_str(),
          signature.params[i].type_name().c_str(), rb_obj_classname(obj), idx + 1);
    }
  }
  return res;
}

inline c10::OptionalArray<int64_t> RubyArgs::intlistOptional(int i) {
  if (NIL_P(args[i])) return {};
  return intlist(i);
}

inline c10::OptionalArray<c10::SymInt> RubyArgs::symintlistOptional(int i) {
  if (NIL_P(args[i])) return {};
  return symintlist(i);
}

inline c10::optional<at::Generator> RubyArgs::generator(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  return Rice::detail::From_Ruby<torch::Generator>().convert(args[i]);
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
    {ID2SYM(rb_intern("complex32")), ScalarType::ComplexHalf},
    {ID2SYM(rb_intern("complex_float")), ScalarType::ComplexFloat},
    {ID2SYM(rb_intern("cfloat")), ScalarType::ComplexFloat},
    {ID2SYM(rb_intern("complex64")), ScalarType::ComplexFloat},
    {ID2SYM(rb_intern("complex_double")), ScalarType::ComplexDouble},
    {ID2SYM(rb_intern("cdouble")), ScalarType::ComplexDouble},
    {ID2SYM(rb_intern("complex128")), ScalarType::ComplexDouble},
    {ID2SYM(rb_intern("bool")), ScalarType::Bool},
    {ID2SYM(rb_intern("qint8")), ScalarType::QInt8},
    {ID2SYM(rb_intern("quint8")), ScalarType::QUInt8},
    {ID2SYM(rb_intern("qint32")), ScalarType::QInt32},
    {ID2SYM(rb_intern("bfloat16")), ScalarType::BFloat16},
  };

  auto it = dtype_map.find(args[i]);
  if (it == dtype_map.end()) {
    rb_raise(rb_eArgError, "invalid dtype: %s", rb_id2name(rb_to_id(args[i])));
  }
  return it->second;
}

inline at::ScalarType RubyArgs::scalartypeWithDefault(int i, at::ScalarType default_scalartype) {
  if (NIL_P(args[i])) return default_scalartype;
  return scalartype(i);
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

inline c10::optional<c10::SymInt> RubyArgs::toSymIntOptional(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  return toSymInt(i);
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
      res[idx] = Rice::detail::From_Ruby<double>().convert(obj);
    } else {
      rb_raise(rb_eArgError, "%s(): argument '%s' must be %s, but found element of type %s at pos %d",
          signature.name.c_str(), signature.params[i].name.c_str(),
          signature.params[i].type_name().c_str(), rb_obj_classname(obj), idx + 1);
    }
  }
  return res;
}

inline at::Layout RubyArgs::layout(int i) {
  if (NIL_P(args[i])) return signature.params[i].default_layout;

  static std::unordered_map<VALUE, Layout> layout_map = {
    {ID2SYM(rb_intern("strided")), Layout::Strided},
  };

  auto it = layout_map.find(args[i]);
  if (it == layout_map.end()) {
    rb_raise(rb_eArgError, "invalid layout: %s", rb_id2name(rb_to_id(args[i])));
  }
  return it->second;
}

inline at::Layout RubyArgs::layoutWithDefault(int i, at::Layout default_layout) {
  if (NIL_P(args[i])) return default_layout;
  return layout(i);
}

inline c10::optional<at::Layout> RubyArgs::layoutOptional(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  return layout(i);
}

inline at::Device RubyArgs::device(int i) {
  if (NIL_P(args[i])) {
    return at::Device("cpu");
  }
  const std::string &device_str = THPUtils_unpackString(args[i]);
  return at::Device(device_str);
}

inline at::Device RubyArgs::deviceWithDefault(int i, const at::Device& default_device) {
  if (NIL_P(args[i])) return default_device;
  return device(i);
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
  return stringWithDefault(i, signature.params[i].default_string);
}

inline std::string RubyArgs::stringWithDefault(int i, const std::string& default_str) {
  if (!args[i]) return default_str;
  return Rice::detail::From_Ruby<std::string>().convert(args[i]);
}

inline c10::optional<std::string> RubyArgs::stringOptional(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  return Rice::detail::From_Ruby<std::string>().convert(args[i]);
}

// string_view does not own data
inline c10::string_view RubyArgs::stringView(int i) {
  return c10::string_view(RSTRING_PTR(args[i]), RSTRING_LEN(args[i]));
}

// string_view does not own data
inline c10::optional<c10::string_view> RubyArgs::stringViewOptional(int i) {
  if (NIL_P(args[i])) return c10::nullopt;
  return c10::string_view(RSTRING_PTR(args[i]), RSTRING_LEN(args[i]));
}

inline int64_t RubyArgs::toInt64(int i) {
  if (NIL_P(args[i])) return signature.params[i].default_int;
  return Rice::detail::From_Ruby<int64_t>().convert(args[i]);
}

inline c10::SymInt RubyArgs::toSymInt(int i) {
  if (NIL_P(args[i])) {
    return c10::SymInt(signature.params[i].default_int);
  }

  // TODO improve
  return c10::SymInt(toInt64(i));
}

inline double RubyArgs::toDouble(int i) {
  if (NIL_P(args[i])) return signature.params[i].default_double;
  return Rice::detail::From_Ruby<double>().convert(args[i]);
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
        rb_raise(rb_eArgError, "RubyArgParser: dst ParsedArgs buffer does not have enough capacity, expected %d (got %d)", static_cast<int>(max_args), N);
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
