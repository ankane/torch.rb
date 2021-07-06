#include <torch/torch.h>

#include <rice/rice.hpp>

#include "special_functions.h"
#include "templates.h"
#include "utils.h"

void init_special(Rice::Module& m) {
  auto rb_mSpecial = Rice::define_module_under(m, "Special");
  rb_mSpecial.add_handler<torch::Error>(handle_error);
  add_special_functions(rb_mSpecial);
}
