#include <torch/torch.h>

#include <rice/rice.hpp>

#include <fstream>

#include "torch_functions.h"
#include "templates.h"
#include "utils.h"

template<typename T>
torch::Tensor make_tensor(Rice::Array a, const std::vector<int64_t> &size, const torch::TensorOptions &options) {
  std::vector<T> vec;
  for (long i = 0; i < a.size(); i++) {
    vec.push_back(Rice::detail::From_Ruby<T>().convert(a[i].value()));
  }

  // hack for requires_grad error
  auto requires_grad = options.requires_grad();
  torch::Tensor t = torch::tensor(vec, options.requires_grad(c10::nullopt));
  if (requires_grad) {
    t.set_requires_grad(true);
  }

  return t.reshape(size);
}

void init_torch(Rice::Module& m) {
  register_handler<torch::Error>(handle_global_error);
  add_torch_functions(m);
  m.define_singleton_function(
      "grad_enabled?",
      []() {
        return torch::GradMode::is_enabled();
      })
    .define_singleton_function(
      "_set_grad_enabled",
      [](bool enabled) {
        torch::GradMode::set_enabled(enabled);
      })
    .define_singleton_function(
      "manual_seed",
      [](uint64_t seed) {
        return torch::manual_seed(seed);
      })
    // config
    .define_singleton_function(
      "show_config",
      []() {
        return torch::show_config();
      })
    .define_singleton_function(
      "parallel_info",
      []() {
        return torch::get_parallel_info();
      })
    // begin operations
    .define_singleton_function(
      "_save",
      [](const torch::IValue &value) {
        auto v = torch::pickle_save(value);
        return Object(rb_str_new(v.data(), v.size()));
      })
    .define_singleton_function(
      "_load",
      [](const std::string &filename) {
        // https://github.com/pytorch/pytorch/issues/20356#issuecomment-567663701
        std::ifstream input(filename, std::ios::binary);
        std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));
        input.close();
        return torch::pickle_load(bytes);
      })
    .define_singleton_function(
      "_from_blob",
      [](Rice::String s, const std::vector<int64_t> &size, const torch::TensorOptions &options) {
        void *data = const_cast<char *>(s.c_str());
        return torch::from_blob(data, size, options);
      })
    .define_singleton_function(
      "_tensor",
      [](Rice::Array a, const std::vector<int64_t> &size, const torch::TensorOptions &options) {
        auto dtype = options.dtype();
        if (dtype == torch::kByte) {
          return make_tensor<uint8_t>(a, size, options);
        } else if (dtype == torch::kChar) {
          return make_tensor<int8_t>(a, size, options);
        } else if (dtype == torch::kShort) {
          return make_tensor<int16_t>(a, size, options);
        } else if (dtype == torch::kInt) {
          return make_tensor<int32_t>(a, size, options);
        } else if (dtype == torch::kLong) {
          return make_tensor<int64_t>(a, size, options);
        } else if (dtype == torch::kFloat) {
          return make_tensor<float>(a, size, options);
        } else if (dtype == torch::kDouble) {
          return make_tensor<double>(a, size, options);
        } else if (dtype == torch::kBool) {
          return make_tensor<uint8_t>(a, size, options);
        } else if (dtype == torch::kComplexFloat) {
          return make_tensor<c10::complex<float>>(a, size, options);
        } else if (dtype == torch::kComplexDouble) {
          return make_tensor<c10::complex<double>>(a, size, options);
        } else {
          throw std::runtime_error("Unsupported type");
        }
      });
}
