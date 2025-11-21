#include <torch/torch.h>

#include <rice/rice.hpp>

void init_fft(Rice::Module& m);
void init_linalg(Rice::Module& m);
void init_nn(Rice::Module& m);
void init_special(Rice::Module& m);
void init_accelerator(Rice::Module& m);
void init_distributed(Rice::Module& m);
void init_tensor(Rice::Module& m, Rice::Class& c, Rice::Class& rb_cTensorOptions);
void init_torch(Rice::Module& m);

void init_backends(Rice::Module& m);
void init_cuda(Rice::Module& m);
void init_device(Rice::Module& m);
void init_generator(Rice::Module& m, Rice::Class& rb_cGenerator);
void init_ivalue(Rice::Module& m, Rice::Class& rb_cIValue);
void init_random(Rice::Module& m);

VALUE rb_eTorchError = Qnil;

extern "C"
void Init_ext() {
  auto m = Rice::define_module("Torch");

  rb_eTorchError = Rice::define_class_under(m, "Error", rb_eStandardError);

  // need to define certain classes up front to keep Rice happy
  auto rb_cIValue = Rice::define_class_under<torch::IValue>(m, "IValue")
    .define_constructor(Rice::Constructor<torch::IValue>());
  auto rb_cGenerator = Rice::define_class_under<torch::Generator>(m, "Generator");
  auto rb_cTensor = Rice::define_class_under<torch::Tensor>(m, "Tensor");
  auto rb_cTensorOptions = Rice::define_class_under<torch::TensorOptions>(m, "TensorOptions")
    .define_constructor(Rice::Constructor<torch::TensorOptions>());

  // keep this order
  init_torch(m);
  init_device(m);
  init_tensor(m, rb_cTensor, rb_cTensorOptions);
  init_nn(m);
  init_fft(m);
  init_linalg(m);
  init_special(m);
  init_accelerator(m);

  init_backends(m);
  init_cuda(m);
  init_generator(m, rb_cGenerator);
  init_ivalue(m, rb_cIValue);
  init_random(m);
  init_distributed(m);
}
