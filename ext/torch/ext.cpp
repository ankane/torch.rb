#include <torch/torch.h>

#include <rice/rice.hpp>

void init_nn(Rice::Module& m);
void init_tensor(Rice::Module& m, Rice::Class& c, Rice::Class& rb_cTensorOptions);
void init_torch(Rice::Module& m);

void init_cuda(Rice::Module& m);
void init_device(Rice::Module& m);
void init_ivalue(Rice::Module& m, Rice::Class& rb_cIValue);
void init_random(Rice::Module& m);

extern "C"
void Init_ext()
{
  auto m = Rice::define_module("Torch");

  // need to define certain classes up front to keep Rice happy
  auto rb_cIValue = Rice::define_class_under<torch::IValue>(m, "IValue")
    .define_constructor(Rice::Constructor<torch::IValue>());
  auto rb_cTensor = Rice::define_class_under<torch::Tensor>(m, "Tensor");
  auto rb_cTensorOptions = Rice::define_class_under<torch::TensorOptions>(m, "TensorOptions")
    .define_constructor(Rice::Constructor<torch::TensorOptions>());

  // keep this order
  init_torch(m);
  init_tensor(m, rb_cTensor, rb_cTensorOptions);
  init_nn(m);

  init_cuda(m);
  init_device(m);
  init_ivalue(m, rb_cIValue);
  init_random(m);
}
