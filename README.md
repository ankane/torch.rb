# Torch.rb

:fire: Deep learning for Ruby, powered by [LibTorch](https://pytorch.org)

Check out:

- [TorchVision](https://github.com/ankane/torchvision) for computer vision tasks
- [TorchText](https://github.com/ankane/torchtext) for text and NLP tasks
- [TorchAudio](https://github.com/ankane/torchaudio) for audio tasks
- [TorchRec](https://github.com/ankane/torchrec-ruby) for recommendation systems
- [TorchData](https://github.com/ankane/torchdata-ruby) for data loading

[![Build Status](https://github.com/ankane/torch.rb/actions/workflows/build.yml/badge.svg)](https://github.com/ankane/torch.rb/actions)

## Installation

First, [download LibTorch](https://pytorch.org/get-started/locally/). For Mac arm64, use:

```sh
curl -L https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.0.zip > libtorch.zip
unzip -q libtorch.zip
```

For Linux x86-64, use the build that matches your CUDA version. For other platforms, build LibTorch from source.

Then run:

```sh
bundle config build.torch-rb --with-torch-dir=/path/to/libtorch
```

And add this line to your application’s Gemfile:

```ruby
gem "torch-rb"
```

It can take 5-10 minutes to compile the extension. Windows is not currently supported.

## Getting Started

A good place to start is [Deep Learning with Torch.rb: A 60 Minute Blitz](tutorials/blitz/README.md).

## Tutorials

- [Transfer learning](tutorials/transfer_learning/README.md)
- [Sequence models](tutorials/nlp/sequence_models.md)
- [Word embeddings](tutorials/nlp/word_embeddings.md)

## Examples

- [Image classification with MNIST](examples/mnist) ([日本語版](https://qiita.com/kojix2/items/c19c36dc1bf73ea93409))
- [Collaborative filtering with MovieLens](examples/movielens)
- [Generative adversarial networks](examples/gan)

## API

This library follows the [PyTorch API](https://pytorch.org/docs/stable/torch.html). There are a few changes to make it more Ruby-like:

- Methods that perform in-place modifications end with `!` instead of `_` (`add!` instead of `add_`)
- Methods that return booleans use `?` instead of `is_`  (`tensor?` instead of `is_tensor`)
- Numo is used instead of NumPy (`x.numo` instead of `x.numpy()`)

You can follow PyTorch tutorials and convert the code to Ruby in many cases. Feel free to open an issue if you run into problems.

## Overview

Some examples below are from [Deep Learning with PyTorch: A 60 Minutes Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

### Tensors

Create a tensor from a Ruby array

```ruby
x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
```

Get the shape of a tensor

```ruby
x.shape
```

There are [many functions](#tensor-creation) to create tensors, like

```ruby
a = Torch.rand(3)
b = Torch.zeros(2, 3)
```

Each tensor has four properties

- `dtype` - the data type - `:uint8`, `:int8`, `:int16`, `:int32`, `:int64`, `:float32`, `:float64`, or `:bool`
- `layout` - `:strided` (dense) or `:sparse`
- `device` - the compute device, like CPU or GPU
- `requires_grad` - whether or not to record gradients

You can specify properties when creating a tensor

```ruby
Torch.rand(2, 3, dtype: :float64, layout: :strided, device: "cpu", requires_grad: true)
```

### Operations

Create a tensor

```ruby
x = Torch.tensor([10, 20, 30])
```

Add

```ruby
x + 5 # tensor([15, 25, 35])
```

Subtract

```ruby
x - 5 # tensor([5, 15, 25])
```

Multiply

```ruby
x * 5 # tensor([50, 100, 150])
```

Divide

```ruby
x / 5 # tensor([2, 4, 6])
```

Get the remainder

```ruby
x % 3 # tensor([1, 2, 0])
```

Raise to a power

```ruby
x**2 # tensor([100, 400, 900])
```

Perform operations with other tensors

```ruby
y = Torch.tensor([1, 2, 3])
x + y # tensor([11, 22, 33])
```

Perform operations in-place

```ruby
x.add!(5)
x # tensor([15, 25, 35])
```

You can also specify an output tensor

```ruby
result = Torch.empty(3)
Torch.add(x, y, out: result)
result # tensor([15, 25, 35])
```

### Numo

Convert a tensor to a Numo array

```ruby
a = Torch.ones(5)
a.numo
```

Convert a Numo array to a tensor

```ruby
b = Numo::NArray.cast([1, 2, 3])
Torch.from_numo(b)
```

### Autograd

Create a tensor with `requires_grad: true`

```ruby
x = Torch.ones(2, 2, requires_grad: true)
```

Perform operations

```ruby
y = x + 2
z = y * y * 3
out = z.mean
```

Backprop

```ruby
out.backward
```

Get gradients

```ruby
x.grad # tensor([[4.5, 4.5], [4.5, 4.5]])
```

Stop autograd from tracking history

```ruby
x.requires_grad # true
(x**2).requires_grad # true

Torch.no_grad do
  (x**2).requires_grad # false
end
```

### Neural Networks

Define a neural network

```ruby
class MyNet < Torch::NN::Module
  def initialize
    super()
    @conv1 = Torch::NN::Conv2d.new(1, 6, 3)
    @conv2 = Torch::NN::Conv2d.new(6, 16, 3)
    @fc1 = Torch::NN::Linear.new(16 * 6 * 6, 120)
    @fc2 = Torch::NN::Linear.new(120, 84)
    @fc3 = Torch::NN::Linear.new(84, 10)
  end

  def forward(x)
    x = Torch::NN::F.max_pool2d(Torch::NN::F.relu(@conv1.call(x)), [2, 2])
    x = Torch::NN::F.max_pool2d(Torch::NN::F.relu(@conv2.call(x)), 2)
    x = Torch.flatten(x, 1)
    x = Torch::NN::F.relu(@fc1.call(x))
    x = Torch::NN::F.relu(@fc2.call(x))
    @fc3.call(x)
  end
end
```

Create an instance of it

```ruby
net = MyNet.new
input = Torch.randn(1, 1, 32, 32)
net.call(input)
```

Get trainable parameters

```ruby
net.parameters
```

Zero the gradient buffers and backprop with random gradients

```ruby
net.zero_grad
out.backward(Torch.randn(1, 10))
```

Define a loss function

```ruby
output = net.call(input)
target = Torch.randn(10)
target = target.view(1, -1)
criterion = Torch::NN::MSELoss.new
loss = criterion.call(output, target)
```

Backprop

```ruby
net.zero_grad
p net.conv1.bias.grad
loss.backward
p net.conv1.bias.grad
```

Update the weights

```ruby
learning_rate = 0.01
net.parameters.each do |f|
  f.data.sub!(f.grad.data * learning_rate)
end
```

Use an optimizer

```ruby
optimizer = Torch::Optim::SGD.new(net.parameters, lr: 0.01)
optimizer.zero_grad
output = net.call(input)
loss = criterion.call(output, target)
loss.backward
optimizer.step
```

### Saving and Loading Models

Save a model

```ruby
Torch.save(net.state_dict, "net.pth")
```

Load a model

```ruby
net = MyNet.new
net.load_state_dict(Torch.load("net.pth"))
net.eval
```

When saving a model in Python to load in Ruby, convert parameters to tensors (due to outstanding bugs in LibTorch)

```python
state_dict = {k: v.data if isinstance(v, torch.nn.Parameter) else v for k, v in state_dict.items()}
torch.save(state_dict, "net.pth")
```

### Tensor Creation

Here’s a list of functions to create tensors (descriptions from the [C++ docs](https://pytorch.org/cppdocs/notes/tensor_creation.html)):

- `arange` returns a tensor with a sequence of integers

  ```ruby
  Torch.arange(3) # tensor([0, 1, 2])
  ```

- `empty` returns a tensor with uninitialized values

  ```ruby
  Torch.empty(3) # tensor([7.0054e-45, 0.0000e+00, 0.0000e+00])
  ```

- `eye` returns an identity matrix

  ```ruby
  Torch.eye(2) # tensor([[1, 0], [0, 1]])
  ```

- `full` returns a tensor filled with a single value

  ```ruby
  Torch.full([3], 5) # tensor([5, 5, 5])
  ```

- `linspace` returns a tensor with values linearly spaced in some interval

  ```ruby
  Torch.linspace(0, 10, 5) # tensor([0, 5, 10])
  ```

- `logspace` returns a tensor with values logarithmically spaced in some interval

  ```ruby
  Torch.logspace(0, 10, 5) # tensor([1, 1e5, 1e10])
  ```

- `ones` returns a tensor filled with all ones

  ```ruby
  Torch.ones(3) # tensor([1, 1, 1])
  ```

- `rand` returns a tensor filled with values drawn from a uniform distribution on [0, 1)

  ```ruby
  Torch.rand(3) # tensor([0.5444, 0.8799, 0.5571])
  ```

- `randint` returns a tensor with integers randomly drawn from an interval

  ```ruby
  Torch.randint(1, 10, [3]) # tensor([7, 6, 4])
  ```

- `randn` returns a tensor filled with values drawn from a unit normal distribution

  ```ruby
  Torch.randn(3) # tensor([-0.7147,  0.6614,  1.1453])
  ```

- `randperm` returns a tensor filled with a random permutation of integers in some interval

  ```ruby
  Torch.randperm(3) # tensor([2, 0, 1])
  ```

- `zeros` returns a tensor filled with all zeros

  ```ruby
  Torch.zeros(3) # tensor([0, 0, 0])
  ```

## LibTorch Compatibility

Here’s the list of compatible versions.

Torch.rb | LibTorch
--- | ---
0.20.x | 2.7.x
0.19.x | 2.6.x
0.18.x | 2.5.x
0.17.x | 2.4.x
0.16.x | 2.3.x
0.15.x | 2.2.x
0.14.x | 2.1.x
0.13.x | 2.0.x
0.12.x | 1.13.x

## Performance

Deep learning is significantly faster on a GPU.

### Linux

With Linux, install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) and reinstall the gem.

Check if CUDA is available

```ruby
Torch::CUDA.available?
```

Move a neural network to a GPU

```ruby
net.cuda
```

If you don’t have a GPU that supports CUDA, we recommend using a cloud service. [Paperspace](https://www.paperspace.com/) has a great free plan. We’ve put together a [Docker image](https://github.com/ankane/ml-stack) to make it easy to get started. On Paperspace, create a notebook with a custom container. Under advanced options, set the container name to:

```text
ankane/ml-stack:torch-gpu
```

And leave the other fields in that section blank. Once the notebook is running, you can run the [MNIST example](https://github.com/ankane/ml-stack/blob/master/torch-gpu/MNIST.ipynb).

### Mac

With Apple silicon, check if Metal Performance Shaders (MPS) is available

```ruby
Torch::Backends::MPS.available?
```

Move a neural network to a GPU

```ruby
device = Torch.device("mps")
net.to(device)
```

## History

View the [changelog](https://github.com/ankane/torch.rb/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/torch.rb/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/torch.rb/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/torch.rb.git
cd torch.rb
bundle install
bundle exec rake compile -- --with-torch-dir=/path/to/libtorch
bundle exec rake test
```

You can use [this script](https://gist.github.com/ankane/9b2b5fcbd66d6e4ccfeb9d73e529abe7) to test on GPUs with the AWS Deep Learning Base AMI (Ubuntu 18.04).

Here are some good resources for contributors:

- [PyTorch API](https://pytorch.org/docs/stable/torch.html)
- [PyTorch C++ API](https://pytorch.org/cppdocs/)
- [Tensor Creation API](https://pytorch.org/cppdocs/notes/tensor_creation.html)
- [Using the PyTorch C++ Frontend](https://pytorch.org/tutorials/advanced/cpp_frontend.html)
