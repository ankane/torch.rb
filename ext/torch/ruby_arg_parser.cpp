// adapted from PyTorch - python_arg_parser.cpp

#include "ruby_arg_parser.hpp"

VALUE THPVariableClass = Qnil;

static std::unordered_map<std::string, ParameterType> type_map = {
  {"Tensor", ParameterType::TENSOR},
  {"Scalar", ParameterType::SCALAR},
  {"int64_t", ParameterType::INT64},
  {"double", ParameterType::DOUBLE},
  {"complex", ParameterType::COMPLEX},
  {"TensorList", ParameterType::TENSOR_LIST},
  {"IntArrayRef", ParameterType::INT_LIST},
  {"ArrayRef<double>", ParameterType::FLOAT_LIST},
  {"Generator", ParameterType::GENERATOR},
  {"bool", ParameterType::BOOL},
  {"Storage", ParameterType::STORAGE},
  // {"PyObject*", ParameterType::PYOBJECT},
  {"ScalarType", ParameterType::SCALARTYPE},
  {"Layout", ParameterType::LAYOUT},
  {"MemoryFormat", ParameterType::MEMORY_FORMAT},
  {"QScheme", ParameterType::QSCHEME},
  {"Device", ParameterType::DEVICE},
  {"std::string", ParameterType::STRING},
  {"Dimname", ParameterType::DIMNAME},
  {"DimnameList", ParameterType::DIMNAME_LIST},
};

static const std::unordered_map<std::string, std::vector<std::string>> numpy_compatibility_arg_names = {
  {"dim", {"axis"}},
  {"keepdim", {"keepdims"}},
  {"input", {"x", "a", "x1"}},
  {"other", {"x2"}},
};

static bool should_allow_numbers_as_tensors(const std::string& name) {
  static std::unordered_set<std::string> allowed = {
    "add", "add_", "add_out",
    "div", "div_", "div_out",
    "mul", "mul_", "mul_out",
    "sub", "sub_", "sub_out",
    "true_divide", "true_divide_", "true_divide_out",
    "floor_divide", "floor_divide_", "floor_divide_out"
  };
  return allowed.find(name) != allowed.end();
}

FunctionParameter::FunctionParameter(const std::string& fmt, bool keyword_only)
  : optional(false)
  , allow_none(false)
  , keyword_only(keyword_only)
  , size(0)
  , default_scalar(0)
{
  auto space = fmt.find(' ');
  if (space == std::string::npos) {
    throw std::runtime_error("FunctionParameter(): missing type: " + fmt);
  }

  auto type_str = fmt.substr(0, space);

  auto question = type_str.find('?');
  if (question != std::string::npos) {
    allow_none = true;
    type_str = type_str.substr(0, question);
  }

  // Parse and remove brackets from type_str
  auto bracket = type_str.find('[');
  if (bracket != std::string::npos) {
    auto size_str = type_str.substr(bracket + 1, type_str.length() - bracket - 2);
    size = atoi(size_str.c_str());
    type_str = type_str.substr(0, bracket);
  }

  auto name_str = fmt.substr(space + 1);
  auto it = type_map.find(type_str);
  if (it == type_map.end()) {
    throw std::runtime_error("FunctionParameter(): invalid type string: " + type_str);
  }
  type_ = it->second;

  auto eq = name_str.find('=');
  if (eq != std::string::npos) {
    name = name_str.substr(0, eq);
    optional = true;
    set_default_str(name_str.substr(eq + 1));
  } else {
    name = name_str;
  }
  ruby_name = THPUtils_internSymbol(name);
  auto np_compat_it = numpy_compatibility_arg_names.find(name);
  if (np_compat_it != numpy_compatibility_arg_names.end()) {
    for (const auto& str: np_compat_it->second) {
      numpy_python_names.push_back(THPUtils_internSymbol(str));
    }
  }
}

bool is_tensor_list(VALUE obj, int argnum, bool throw_error) {
  if (!RB_TYPE_P(obj, T_ARRAY)) {
    return false;
  }
  auto size = RARRAY_LEN(obj);
  for (int idx = 0; idx < size; idx++) {
    VALUE iobj = rb_ary_entry(obj, idx);
    if (!THPVariable_Check(iobj)) {
      if (throw_error) {
        throw Exception(rb_eArgError, "expected Tensor as element %d in argument %d, but got %s",
            static_cast<int>(idx), argnum, rb_obj_classname(obj));
      }
      return false;
    }
  }
  return true;
}

// argnum is needed for raising the TypeError, it's used in the error message.
auto FunctionParameter::check(VALUE obj, int argnum) -> bool
{
  switch (type_) {
    case ParameterType::TENSOR: {
      if (THPVariable_Check(obj)) {
        return true;
      }
      return allow_numbers_as_tensors && THPUtils_checkScalar(obj);
    }
    case ParameterType::SCALAR:
    case ParameterType::COMPLEX:
      if (RB_TYPE_P(obj, T_COMPLEX)) {
        return true;
      }
      // fallthrough
    case ParameterType::DOUBLE: {
      if (RB_FLOAT_TYPE_P(obj) || FIXNUM_P(obj)) {
        return true;
      }
      if (THPVariable_Check(obj)) {
        auto var = from_ruby<torch::Tensor>(obj);
        return !var.requires_grad() && var.dim() == 0;
      }
      return false;
    }
    case ParameterType::INT64: {
      if (FIXNUM_P(obj)) {
        return true;
      }
      if (THPVariable_Check(obj)) {
        auto var = from_ruby<torch::Tensor>(obj);
        return at::isIntegralType(var.scalar_type(), /*includeBool=*/false) && !var.requires_grad() && var.dim() == 0;
      }
      return false;
    }
    // case ParameterType::DIMNAME: return THPUtils_checkDimname(obj);
    // case ParameterType::DIMNAME_LIST: {
    //   if (THPUtils_checkDimnameList(obj)) {
    //     return true;
    //   }
    //   // if a size is specified (e.g. DimnameList[1]) we also allow passing a single Dimname
    //   return size == 1 && THPUtils_checkDimname(obj);
    // }
    case ParameterType::TENSOR_LIST: {
      return is_tensor_list(obj, argnum, true /* throw_error */);
    }
    case ParameterType::INT_LIST: {
      if (RB_TYPE_P(obj, T_ARRAY)) {
        return true;
      }
      // if a size is specified (e.g. IntArrayRef[2]) we also allow passing a single int
      return size > 0 && FIXNUM_P(obj);
    }
    case ParameterType::FLOAT_LIST: return (RB_TYPE_P(obj, T_ARRAY));
    // case ParameterType::GENERATOR: return THPGenerator_Check(obj);
    case ParameterType::BOOL: return obj == Qtrue || obj == Qfalse;
    // case ParameterType::STORAGE: return isStorage(obj);
    // case ParameterType::PYOBJECT: return true;
    // case ParameterType::SCALARTYPE: return THPDtype_Check(obj) || THPPythonScalarType_Check(obj);
    // case ParameterType::LAYOUT: return THPLayout_Check(obj);
    // case ParameterType::MEMORY_FORMAT: return THPMemoryFormat_Check(obj);
    // case ParameterType::QSCHEME: return THPQScheme_Check(obj);
    // case ParameterType::DEVICE:
    //   return FIXNUM_P(obj) || SYMBOL_P(obj) || THPDevice_Check(obj);
    case ParameterType::STRING: return RB_TYPE_P(obj, T_STRING);
    default: throw std::runtime_error("unknown parameter type");
  }
}

std::string FunctionParameter::type_name() const {
  switch (type_) {
    case ParameterType::TENSOR: return "Tensor";
    case ParameterType::SCALAR: return "Number";
    case ParameterType::INT64: return "int";
    case ParameterType::DOUBLE: return "float";
    case ParameterType::COMPLEX: return "complex";
    case ParameterType::TENSOR_LIST: return "array of Tensors";
    case ParameterType::INT_LIST: return "array of ints";
    case ParameterType::FLOAT_LIST: return "array of floats";
    case ParameterType::GENERATOR: return "torch.Generator";
    case ParameterType::BOOL: return "bool";
    case ParameterType::STORAGE: return "torch.Storage";
    // case ParameterType::PYOBJECT: return "object";
    case ParameterType::SCALARTYPE: return "torch.dtype";
    case ParameterType::LAYOUT: return "torch.layout";
    case ParameterType::MEMORY_FORMAT: return "torch.memory_format";
    case ParameterType::QSCHEME: return "torch.qscheme";
    case ParameterType::DEVICE: return "torch.device";
    case ParameterType::STRING: return "str";
    case ParameterType::DIMNAME: return "name";
    case ParameterType::DIMNAME_LIST: return "array of names";
    default: throw std::runtime_error("unknown parameter type");
  }
}

static inline c10::optional<int64_t> parse_as_integer(const std::string& s) {
  if (s.empty())
    return c10::nullopt;
  char *str_end;
  long ans = strtol(s.c_str(), &str_end, 0);
  // *str_end == 0 if the entire string was parsed as an integer.
  return (*str_end == 0) ? c10::optional<int64_t>(ans) : c10::nullopt;
}

/*
Parse default value of IntArrayRef declared at native_functions.yaml

There are two kinds of default values:
1. IntArrayRef[2] x=1 (where size=2, value={1,1}
2. IntArrayRef x={1,2,3} (where size=3, value={1,2,3}, note that there cannot be space after comma since native_parse.py uses ', ' to split args)
*/
static inline std::vector<int64_t> parse_intlist_args(const std::string& s, int64_t size) {
  size_t n = s.size();

  if (s.empty()) return std::vector<int64_t>();

  // case 1. s is an int (e.g., s=2)
  if (s[0] != '{') {
    return std::vector<int64_t>(size, std::stol(s));
  }

  // case 2. s is a list of dims (e.g., s={1,2})

  // since already checked left brace '{' above, here only checks right brace '}'
  TORCH_CHECK(s[n - 1] == '}', "Default value of IntArrayRef is missing right brace '}', found ", s[n - 1]);

  auto args = std::vector<int64_t>();
  std::istringstream ss(s.substr(1, s.length() - 2)); // exclude '{' and '}'
  std::string tok;

  while(std::getline(ss, tok, ',')) {
    args.emplace_back(std::stol(tok));
  }
  return args;
}

void FunctionParameter::set_default_str(const std::string& str) {
  if (str == "None") {
    allow_none = true;
  }
  if (type_ == ParameterType::TENSOR) {
    if (str != "None") {
      throw std::runtime_error("default value for Tensor must be none, got: " + str);
    }
  } else if (type_ == ParameterType::INT64) {
    default_int = atol(str.c_str());
  } else if (type_ == ParameterType::BOOL) {
    default_bool = (str == "True" || str == "true");
  } else if (type_ == ParameterType::DOUBLE) {
    default_double = atof(str.c_str());
  } else if (type_ == ParameterType::COMPLEX) {
    default_complex[0] = atof(str.c_str()); // TODO: parse "x + xj"?
    default_complex[1] = 0;
  } else if (type_ == ParameterType::SCALAR) {
    if (str != "None") {
      // we sometimes rely on integer-vs-float values, e.g. with arange.
      const auto as_integer = parse_as_integer(str);
      default_scalar = as_integer.has_value() ? at::Scalar(as_integer.value()) :
                                                at::Scalar(atof(str.c_str()));
    }
  } else if (type_ == ParameterType::INT_LIST) {
    if (str != "None") {
      default_intlist = parse_intlist_args(str, size);
    }
  } else if (type_ == ParameterType::FLOAT_LIST) {
    if (str != "None") {
      throw std::runtime_error("Defaults not supported for float[]");
    }
  } else if (type_ == ParameterType::SCALARTYPE) {
    if (str == "None") {
      default_scalartype = at::ScalarType::Undefined;
    } else if (str == "torch.int64") {
      default_scalartype = at::ScalarType::Long;
    } else {
      throw std::runtime_error("invalid default value for ScalarType: " + str);
    }
  } else if (type_ == ParameterType::LAYOUT) {
    if (str == "None") {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(allow_none);
    } else if (str == "torch.strided") {
      default_layout = at::Layout::Strided;
    } else if (str == "torch.sparse_coo") {
      default_layout = at::Layout::Sparse;
    } else {
      throw std::runtime_error("invalid default value for layout: " + str);
    }
  } else if (type_ == ParameterType::DEVICE) {
    if (str != "None") {
      throw std::runtime_error("invalid device: " + str);
    }
  } else if (type_ == ParameterType::STRING) {
    if (str != "None" && str != "") {
      throw std::runtime_error("invalid default string: " + str);
    }
  }
}

FunctionSignature::FunctionSignature(const std::string& fmt, int index)
  : min_args(0)
  , max_args(0)
  , max_pos_args(0)
  , index(index)
  , hidden(false)
  , deprecated(false)
{
  auto open_paren = fmt.find('(');
  if (open_paren == std::string::npos) {
    throw std::runtime_error("missing opening parenthesis: " + fmt);
  }
  name = fmt.substr(0, open_paren);

  bool allow_numbers_as_tensors = should_allow_numbers_as_tensors(name);

  auto last_offset = open_paren + 1;
  auto next_offset = last_offset;
  bool keyword_only = false;
  bool done = false;
  while (!done) {
    auto offset = fmt.find(", ", last_offset);
    if (offset == std::string::npos) {
      offset = fmt.find(')', last_offset);
      done = true;
      next_offset = offset+ 1;
      // this 'if' happens for an empty parameter list, i.e. fn().
      if (offset == last_offset) {
        last_offset = next_offset;
        break;
      }
    } else {
      next_offset = offset + 2;
    }
    if (offset == std::string::npos) {
      throw std::runtime_error("missing closing parenthesis: " + fmt);
    }
    if (offset == last_offset) {
      throw std::runtime_error("malformed signature: " + fmt);
    }

    auto param_str = fmt.substr(last_offset, offset - last_offset);
    last_offset = next_offset;
    if (param_str == "*") {
      keyword_only = true;
    } else {
      params.emplace_back(param_str, keyword_only);
      params.back().allow_numbers_as_tensors = allow_numbers_as_tensors;
    }
  }

  if (fmt.substr(last_offset) == "|deprecated") {
    hidden = true;
    // TODO: raise warning when parsing deprecated signatures
    deprecated = true;
  } else if (fmt.substr(last_offset) == "|hidden") {
    hidden = true;
  }

  max_args = params.size();

  // count the number of non-optional args
  for (auto& param : params) {
    if (!param.optional) {
      min_args++;
    }
    if (!param.keyword_only) {
      max_pos_args++;
    }
  }
}

std::string FunctionSignature::toString() const {
  // TODO: consider printing more proper schema strings with defaults, optionals, etc.
  std::ostringstream ss;
  bool keyword_already = false;
  ss << "(";
  int i = 0;
  for (auto& param : params) {
    if (i != 0) {
      ss << ", ";
    }
    if (param.keyword_only && !keyword_already) {
      ss << "*, ";
      keyword_already = true;
    }
    ss << param.type_name() << " " << param.name;
    i++;
  }
  ss << ")";
  return ss.str();
}

[[noreturn]]
static void extra_args(const FunctionSignature& signature, ssize_t nargs) {
  const long max_pos_args = signature.max_pos_args;
  const long min_args = signature.min_args;
  const long nargs_ = nargs;
  if (min_args != max_pos_args) {
    throw Exception(rb_eArgError, "%s() takes from %ld to %ld positional arguments but %ld were given",
        signature.name.c_str(), min_args, max_pos_args, nargs_);
  }
  throw Exception(rb_eArgError, "%s() takes %ld positional argument%s but %ld %s given",
      signature.name.c_str(),
      max_pos_args, max_pos_args == 1 ? "" : "s",
      nargs_, nargs == 1 ? "was" : "were");
}

[[noreturn]]
static void missing_args(const FunctionSignature& signature, int idx) {
  int num_missing = 0;
  std::stringstream ss;

  auto& params = signature.params;
  for (auto it = params.begin() + idx; it != params.end(); ++it) {
    if (!it->optional) {
      if (num_missing > 0) {
        ss << ", ";
      }
      ss << '"' << it->name << '"';
      num_missing++;
    }
  }

  throw Exception(rb_eArgError, "%s() missing %d required positional argument%s: %s",
      signature.name.c_str(),
      num_missing,
      num_missing == 1 ? "s" : "",
      ss.str().c_str());
}

static ssize_t find_param(FunctionSignature& signature, VALUE name) {
  ssize_t i = 0;
  for (auto& param : signature.params) {
    bool cmp = name == param.ruby_name;
    if (cmp) {
      return i;
    }
    i++;
  }
  return -1;
}

[[noreturn]]
static void extra_kwargs(FunctionSignature& signature, VALUE kwargs, ssize_t num_pos_args) {
  VALUE key, value;

  VALUE keys = rb_funcall(kwargs, rb_intern("keys"), 0);
  if (RARRAY_LEN(keys) > 0) {
    key = rb_ary_entry(keys, 0);
    value = rb_hash_aref(kwargs, key);

    if (!THPUtils_checkSymbol(key)) {
      throw Exception(rb_eArgError, "keywords must be symbols, not %s", rb_obj_classname(key));
    }

    auto param_idx = find_param(signature, key);
    if (param_idx < 0) {
      throw Exception(rb_eArgError, "%s() got an unexpected keyword argument '%s'",
          signature.name.c_str(), THPUtils_unpackSymbol(key).c_str());
    }

    if (param_idx < num_pos_args) {
      throw Exception(rb_eArgError, "%s() got multiple values for argument '%s'",
          signature.name.c_str(), THPUtils_unpackSymbol(key).c_str());
    }
  }

  // this should never be hit
  throw Exception(rb_eArgError, "invalid keyword arguments");
}

// TODO use Qundef
VALUE missing = Qnil;

bool FunctionSignature::parse(VALUE self, VALUE args, VALUE kwargs, std::vector<VALUE> &dst,  // NOLINT
                              bool raise_exception) {
  auto nargs = NIL_P(args) ? 0 : RARRAY_LEN(args);
  ssize_t remaining_kwargs = NIL_P(kwargs) ? 0 :  RHASH_SIZE(kwargs);
  ssize_t arg_pos = 0;
  bool allow_varargs_intlist = false;

  // if there is a single positional IntArrayRef argument, i.e. expand(..), view(...),
  // allow a var-args style IntArrayRef, so expand(5,3) behaves as expand((5,3))
  if (max_pos_args == 1 && params[0].type_ == ParameterType::INT_LIST) {
    allow_varargs_intlist = true;
  }

  if (nargs > max_pos_args && !allow_varargs_intlist) {
    if (raise_exception) {
      // foo() takes takes 2 positional arguments but 3 were given
      extra_args(*this, nargs);
    }
    return false;
  }

  // if (!overloaded_args.empty()) {
  //   overloaded_args.clear();
  // }

  int i = 0;
  // if (self != nullptr && !THPVariable_CheckExact(self) && check_has_torch_function(self)) {
  //   append_overloaded_arg(&this->overloaded_args, self);
  // }
  for (auto& param : params) {
    VALUE obj = missing;
    bool is_kwd = false;
    if (arg_pos < nargs) {
      // extra positional args given after single positional IntArrayRef arg
      if (param.keyword_only) {
        if (raise_exception) {
          extra_args(*this, nargs);
        }
        return false;
      }
      obj = rb_ary_entry(args, arg_pos);
    } else if (!NIL_P(kwargs)) {
      obj = rb_hash_lookup2(kwargs, param.ruby_name, missing);
      // for (VALUE numpy_name: param.numpy_python_names) {
      //   if (obj) {
      //     break;
      //   }
      //   obj = rb_hash_aref(kwargs, numpy_name);
      // }
      is_kwd = true;
    }

    if ((obj == missing && param.optional) || (NIL_P(obj) && param.allow_none)) {
      dst[i++] = Qnil;
    } else if (obj == missing) {
      if (raise_exception) {
        // foo() missing 1 required positional argument: "b"
        missing_args(*this, i);
      }
      return false;
    } else if (param.check(obj, i)) {
      dst[i++] = obj;
    // XXX: the Variable check is necessary because sizes become tensors when
    // tracer is enabled. This behavior easily leads to ambiguities, and we
    // should avoid having complex signatures that make use of it...
    } else if (allow_varargs_intlist && arg_pos == 0 && !is_kwd &&
               THPUtils_checkIndex(obj)) {
      // take all positional arguments as this parameter
      // e.g. permute(1, 2, 3) -> permute((1, 2, 3))
      dst[i++] = args;
      arg_pos = nargs;
      continue;
    } else if (raise_exception) {
      if (is_kwd) {
        // foo(): argument 'other' must be str, not int
        throw Exception(rb_eArgError, "%s(): argument '%s' must be %s, not %s",
            name.c_str(), param.name.c_str(), param.type_name().c_str(),
            rb_obj_classname(obj));
      } else {
        // foo(): argument 'other' (position 2) must be str, not int
        throw Exception(rb_eArgError, "%s(): argument '%s' (position %ld) must be %s, not %s",
            name.c_str(), param.name.c_str(), static_cast<long>(arg_pos + 1),
            param.type_name().c_str(), rb_obj_classname(obj));
      }
    } else {
      return false;
    }

    if (!is_kwd) {
      arg_pos++;
    } else if (obj != missing) {
      remaining_kwargs--;
    }
  }

  if (remaining_kwargs > 0) {
    if (raise_exception) {
      // foo() got an unexpected keyword argument "b"
      extra_kwargs(*this, kwargs, nargs);
    }
    return false;
  }
  return true;
}
