## 0.2.1 (unreleased)

- Added support for loading tensors
- Reduced gem size

## 0.2.0 (2020-04-22)

- No longer experimental
- Updated libtorch to 1.5.0
- Added support for GPUs and OpenMP
- Added adaptive pooling layers
- Tensor `dtype` is now based on Numo type for `Torch.tensor`
- Improved support for boolean tensors
- Fixed error with unbiased linear model

## 0.1.8 (2020-01-17)

- Updated libtorch to 1.4.0

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
