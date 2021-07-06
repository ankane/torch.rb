#include <torch/torch.h>

#include <rice/rice.hpp>

#include "linalg_functions.h"
#include "templates.h"
#include "utils.h"

void init_linalg(Rice::Module& m) {
  auto rb_mLinalg = Rice::define_module_under(m, "Linalg");
  rb_mLinalg.add_handler<torch::Error>(handle_error);
  add_linalg_functions(rb_mLinalg);
}
