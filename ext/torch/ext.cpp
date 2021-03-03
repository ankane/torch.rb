#include <rice/rice.hpp>

void init_nn(Rice::Module& m);
void init_tensor(Rice::Module& m);
void init_torch(Rice::Module& m);

void init_cuda(Rice::Module& m);
void init_device(Rice::Module& m);
void init_ivalue(Rice::Module& m);
void init_random(Rice::Module& m);

extern "C"
void Init_ext()
{
  auto m = Rice::define_module("Torch");

  // keep this order
  init_torch(m);
  init_tensor(m);
  init_nn(m);

  init_cuda(m);
  init_device(m);
  init_ivalue(m);
  init_random(m);
}
