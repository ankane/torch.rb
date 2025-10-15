## 0.22.0 (2025-10-15)

- Updated LibTorch to 2.9.0
- Improved error classes

## 0.21.0 (2025-08-07)

- Updated LibTorch to 2.8.0
- Dropped support for Ruby < 3.2

## 0.20.0 (2025-04-26)

- Updated LibTorch to 2.7.0
- Added `Normal` distribution
- Fixed `SystemStackError` with certain tensor comparisons

## 0.19.1 (2025-02-10)

- Fixed error with Rice 4.5

## 0.19.0 (2025-01-29)

- Updated LibTorch to 2.6.0
- Improved `inspect` for `Device`
- Fixed equality for `Device`
- Fixed `index` method for `Device` when no index

## 0.18.0 (2024-10-22)

- Updated LibTorch to 2.5.0

## 0.17.1 (2024-08-19)

- Added `persistent` option to `register_buffer` method
- Added `prefix` and `recurse` options to `named_buffers` method

## 0.17.0 (2024-07-26)

- Updated LibTorch to 2.4.0
- Added `normalize` method
- Added support for tensor indexing with arrays

## 0.16.0 (2024-06-12)

- Updated LibTorch to 2.3.0
- Added `ELU` and `GELU` classes
- Dropped support for Ruby < 3.1

## 0.15.0 (2024-02-28)

- Updated LibTorch to 2.2.0
- Fixed error with `inspect` for MPS tensors

## 0.14.1 (2023-12-26)

- Fixed default arguments for `conv1d`

## 0.14.0 (2023-11-09)

- Updated LibTorch to 2.1.0
- Improved performance of saving and loading models

## 0.13.2 (2023-05-11)

- Fixed error on Fedora

## 0.13.1 (2023-05-03)

- Fixed error with Rice 4.1

## 0.13.0 (2023-04-13)

- Updated LibTorch to 2.0.0
- Dropped support for Ruby < 3

## 0.12.2 (2023-01-30)

- Added experimental support for DataPipes

## 0.12.1 (2023-01-29)

- Added `Generator` class

## 0.12.0 (2022-11-05)

- Updated LibTorch to 1.13.0

## 0.11.2 (2022-09-25)

- Improved LibTorch detection for Homebrew on Mac ARM and Linux

## 0.11.1 (2022-07-06)

- Fixed error with `stft` method

## 0.11.0 (2022-07-06)

- Updated LibTorch to 1.12.0
- Dropped support for Ruby < 2.7

## 0.10.2 (2022-06-14)

- Improved numeric operations between scalars and tensors
- Fixed `dtype` of `cumsum` method

## 0.10.1 (2022-04-12)

- Fixed `dtype`, `device`, and `layout` for `new_*` and `like_*` methods

## 0.10.0 (2022-03-13)

- Updated LibTorch to 1.11.0
- Added `ParameterList`

## 0.9.2 (2022-02-03)

- Added support for setting `nil` gradient
- Added checks when setting gradient
- Fixed precision with `Torch.tensor` method
- Fixed memory issue when creating tensor for `ByteStorage`

## 0.9.1 (2022-02-02)

- Moved `like` methods to C++
- Fixed memory issue

## 0.9.0 (2021-10-23)

- Updated LibTorch to 1.10.0
- Added `real` and `imag` methods to tensors

## 0.8.3 (2021-10-17)

- Fixed `dup` method for tensors and parameters
- Fixed issues with transformers

## 0.8.2 (2021-10-03)

- Added transformers
- Added left shift and right shift

## 0.8.1 (2021-06-15)

- Added `Backends` module
- Added `FFT` module
- Added `Linalg` module
- Added `Special` module

## 0.8.0 (2021-06-15)

- Updated LibTorch to 1.9.0

## 0.7.0 (2021-05-23)

- Updated to Rice 4
- Added support for complex numbers

## 0.6.0 (2021-03-25)

- Updated LibTorch to 1.8.0
- Fixed tensor indexing with endless ranges that exclude end
- Removed support for Ruby 2.5

## 0.5.3 (2021-01-14)

- Added `manual_seed` and `manual_seed_all` for CUDA
- Improved saving and loading models
- Fixed error with tensor indexing with beginless ranges

## 0.5.2 (2020-10-29)

- Fixed `undefined symbol` error with CUDA

## 0.5.1 (2020-10-28)

- Fixed error with tensor classes and no arguments
- Fixed error with `stft` and `clamp` methods

## 0.5.0 (2020-10-28)

- Updated LibTorch to 1.7.0
- Removed deprecated overload for `addcmul!` and `addcdiv!`

## 0.4.2 (2020-10-27)

- Fixed errors with optimizer options

## 0.4.1 (2020-10-12)

- Fixed installation error with Ruby < 2.7

## 0.4.0 (2020-09-27)

- Improved performance of methods
- Improved performance of tensor indexing

## 0.3.7 (2020-09-22)

- Improved performance
- Added `Upsample`
- Added support for passing tensor class to `type` method
- Fixed error with buffers on GPU
- Fixed error with `new_full`
- Fixed issue with `numo` method and non-contiguous tensors

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
