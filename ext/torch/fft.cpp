#include <torch/torch.h>

#include <rice/rice.hpp>

#include "fft_functions.h"
#include "templates.h"
#include "utils.h"

void init_fft(Rice::Module& m) {
  auto rb_mFFT = Rice::define_module_under(m, "FFT");
  rb_mFFT.add_handler<torch::Error>(handle_error);
  add_fft_functions(rb_mFFT);
}
