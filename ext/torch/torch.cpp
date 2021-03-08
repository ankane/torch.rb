#include <torch/torch.h>

#include <rice/rice.hpp>

#include "torch_functions.h"
#include "templates.h"
#include "utils.h"

void init_torch(Rice::Module& m) {
  m.add_handler<torch::Error>(handle_error);
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
      *[] {
        return torch::show_config();
      })
    .define_singleton_function(
      "parallel_info",
      *[] {
        return torch::get_parallel_info();
      })
    // begin operations
    .define_singleton_function(
      "_save",
      [](const torch::IValue &value) {
        auto v = torch::pickle_save(value);
        std::string str(v.begin(), v.end());
        return str;
      })
    .define_singleton_function(
      "_load",
      [](const std::string &s) {
        std::vector<char> v;
        std::copy(s.begin(), s.end(), std::back_inserter(v));
        // https://github.com/pytorch/pytorch/issues/20356#issuecomment-567663701
        return torch::pickle_load(v);
      })
    .define_singleton_function(
      "_from_blob",
      [](Rice::String s, std::vector<int64_t> size, const torch::TensorOptions &options) {
        void *data = const_cast<char *>(s.c_str());
        return torch::from_blob(data, size, options);
      })
    .define_singleton_function(
      "_tensor",
      [](Rice::Array a, std::vector<int64_t> size, const torch::TensorOptions &options) {
        auto dtype = options.dtype();
        torch::Tensor t;
        if (dtype == torch::kBool) {
          std::vector<uint8_t> vec;
          for (size_t i = 0; i < a.size(); i++) {
            vec.push_back(Rice::detail::From_Ruby<bool>::convert(a[i].value()));
          }
          t = torch::tensor(vec, options);
        } else {
          std::vector<float> vec;
          for (size_t i = 0; i < a.size(); i++) {
            vec.push_back(Rice::detail::From_Ruby<float>::convert(a[i].value()));
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
}
