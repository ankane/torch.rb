## 0.3.7 (unreleased)

- Fixed error with buffers on GPU

## 0.3.6 (2020-09-17)

- Added `inplace` option for leaky ReLU
- Fixed error with methods that return a tensor list (`chunk`, `split`, and `unbind`)
- Fixed error with buffers on GPU

## 0.3.5 (2020-09-04)

- Fixed error with data loader (due to `dtype` of `randperm`)

## 0.3.4 (2020-08-26)

- Added `Torch.clamp` method

## 0.3.3 (2020-08-25)

- Added spectral ops
- Fixed tensor indexing

## 0.3.2 (2020-08-24)

- Added `enable_grad` method
- Added `random_split` method
- Added `collate_fn` option to `DataLoader`
- Added `grad=` method to `Tensor`
- Fixed error with `grad` method when empty
- Fixed `EmbeddingBag`

## 0.3.1 (2020-08-17)

- Added `create_graph` and `retain_graph` options to `backward` method
- Fixed error when `set` not required

## 0.3.0 (2020-07-29)

- Updated LibTorch to 1.6.0
- Removed `state_dict` method from optimizers until `load_state_dict` is implemented

## 0.2.7 (2020-06-29)

- Made tensors enumerable
- Improved performance of `inspect` method

## 0.2.6 (2020-06-29)

- Added support for indexing with tensors
- Added `contiguous` methods
- Fixed named parameters for nested parameters

## 0.2.5 (2020-06-07)

- Added `download_url_to_file` and `load_state_dict_from_url` to `Torch::Hub`
- Improved error messages
- Fixed tensor slicing

## 0.2.4 (2020-04-29)

- Added `to_i` and `to_f` to tensors
- Added `shuffle` option to data loader
- Fixed `modules` and `named_modules` for nested modules

## 0.2.3 (2020-04-28)

- Added `show_config` and `parallel_info` methods
- Added `initial_seed` and `seed` methods to `Random`
- Improved data loader
- Build with MKL-DNN and NNPACK when available
- Fixed `inspect` for modules

## 0.2.2 (2020-04-27)

- Added support for saving tensor lists
- Added `ndim` and `ndimension` methods to tensors

## 0.2.1 (2020-04-26)

- Added support for saving and loading models
- Improved error messages
- Reduced gem size

## 0.2.0 (2020-04-22)

- No longer experimental
- Updated LibTorch to 1.5.0
- Added support for GPUs and OpenMP
- Added adaptive pooling layers
- Tensor `dtype` is now based on Numo type for `Torch.tensor`
- Improved support for boolean tensors
- Fixed error with unbiased linear model

## 0.1.8 (2020-01-17)

- Updated LibTorch to 1.4.0

## 0.1.7 (2020-01-10)

- Fixed installation error with Ruby 2.7

## 0.1.6 (2019-12-09)

- Added recurrent layers
- Added more pooling layers
- Added normalization layers

## 0.1.5 (2019-12-06)

- Added many more functions
- Added tensor classes - `FloatTensor`, `LongTensor`, etc
- Improved modules

## 0.1.4 (2019-12-01)

- Added distance functions
- Added more activations
- Added more linear layers
- Added more loss functions
- Added more init methods
- Added support for tensor assignment

## 0.1.3 (2019-11-30)

- Changed to BSD 3-Clause license to match PyTorch
- Added many optimizers
- Added `StepLR` learning rate scheduler
- Added dropout
- Added embedding
- Added support for `bool` type
- Improved performance of `from_numo`

## 0.1.2 (2019-11-27)

- Added SGD optimizer
- Added support for gradient to `backward` method
- Added `argmax`, `eq`, `leaky_relu`, `prelu`, and `reshape` methods
- Improved indexing
- Fixed `zero_grad`
- Fixed error with infinite values

## 0.1.1 (2019-11-26)

- Added support for `uint8` and `int8` types
- Fixed `undefined symbol` error on Linux
- Fixed C++ error messages

## 0.1.0 (2019-11-26)

- First release
