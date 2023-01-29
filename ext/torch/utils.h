#pragma once

#include <torch/torch.h>

#include <rice/rice.hpp>
#include <rice/stl.hpp>

static_assert(
  TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR == 13,
  "Incompatible LibTorch version"
);

// TODO find better place
inline void handle_error(torch::Error const & ex) {
  throw Rice::Exception(rb_eRuntimeError, ex.what_without_backtrace());
}

// keep THP prefix for now to make it easier to compare code

extern VALUE THPGeneratorClass;
extern VALUE THPVariableClass;

inline VALUE THPUtils_internSymbol(const std::string& str) {
  return Rice::Symbol(str);
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

inline bool THPUtils_checkLong(VALUE obj) {
  return FIXNUM_P(obj);
}

inline bool THPUtils_checkScalar(VALUE obj) {
  return FIXNUM_P(obj) || RB_FLOAT_TYPE_P(obj) || RB_TYPE_P(obj, T_COMPLEX);
}

inline bool THPGenerator_Check(VALUE obj) {
  return rb_obj_is_kind_of(obj, THPGeneratorClass);
}

inline bool THPVariable_Check(VALUE obj) {
  return rb_obj_is_kind_of(obj, THPVariableClass);
}

inline bool THPVariable_CheckExact(VALUE obj) {
  return rb_obj_is_instance_of(obj, THPVariableClass);
}
