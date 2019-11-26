# Torch-rb

:fire: Deep learning for Ruby, powered by [LibTorch](https://pytorch.org)

**Note:** This gem is currently experimental. There may be breaking changes between each release.

## Installation

First, [install LibTorch](#libtorch-installation). For Homebrew, use:

```sh
brew install ankane/brew/libtorch
```

Add this line to your application’s Gemfile:

```ruby
gem 'torch-rb'
```

## Getting Started

This library follows the [PyTorch API](https://pytorch.org/docs/stable/torch.html). There are a few changes to make it more Ruby-like:

- Methods that perform in-place modifications end with `!` instead of `_` (`add!` instead of `add_`)
- Methods that return booleans use `?` instead of `is_`  (`tensor?` instead of `is_tensor`)
- Numo is used instead of NumPy (`x.numo` instead of `x.numpy()`)

Many methods and options are missing at the moment. PRs welcome!

Some examples below are from [Deep Learning with PyTorch: A 60 Minutes Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html).

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

- `dtype` - the data type - `:uint8`, `:int8`, `:int16`, `:int32`, `:int64`, `:float32`, `float64`, or `:bool`
- `layout` - `:strided` (dense) or `:sparse`
- `device` - the compute device, like CPU or GPU
- `requires_grad` - whether or not to record gradients

You can specify properties when creating a tensor

```ruby
Torch.rand(2, 3, dtype: :double, layout: :strided, device: "cpu", requires_grad: true)
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
Torch.from_numpy(b)
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
x.grad
```

Stop autograd from tracking history

```ruby
x.requires_grad # true
(x ** 2).requires_grad # true

Torch.no_grad do
  (x ** 2).requires_grad # false
end
```

### Neural Networks

Define a neural network

```ruby
class Net < Torch::NN::Module
  def initialize
    super
    @conv1 = Torch::NN::Conv2d.new(1, 6, 3)
    @conv2 = Torch::NN::Conv2d.new(6, 16, 3)
    @fc1 = Torch::NN::Linear.new(16 * 6 * 6, 120)
    @fc2 = Torch::NN::Linear.new(120, 84)
    @fc3 = Torch::NN::Linear.new(84, 10)
  end

  def forward(x)
    x = Torch::NN::F.max_pool2d(Torch::NN::F.relu(@conv1.call(x)), [2, 2])
    x = Torch::NN::F.max_pool2d(Torch::NN::F.relu(@conv2.call(x)), 2)
    x = x.view(-1, num_flat_features(x))
    x = Torch::NN::F.relu(@fc1.call(x))
    x = Torch::NN::F.relu(@fc2.call(x))
    x = @fc3.call(x)
    x
  end

  def num_flat_features(x)
    size = x.size[1..-1]
    num_features = 1
    size.each do |s|
      num_features *= s
    end
    num_features
  end
end
```

And run

```ruby
net = Net.new
input = Torch.randn(1, 1, 32, 32)
net.call(input)
```

### Tensor Creation

Here’s a list of functions to create tensors (descriptions from the [C++ docs](https://pytorch.org/cppdocs/notes/tensor_creation.html)):

- `arange` returns a tensor with a sequence of integers

  ```ruby
  Torch.arange(3) # tensor([0, 1, 2])
  ```

- `empty` returns a tensor with uninitialized values

  ```ruby
  Torch.empty(3)
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
  Torch.rand(3)
  ```

- `randint` returns a tensor with integers randomly drawn from an interval

  ```ruby
  Torch.randint(1, 10, [3])
  ```

- `randn` returns a tensor filled with values drawn from a unit normal distribution

  ```ruby
  Torch.randn(3)
  ```

- `randperm` returns a tensor filled with a random permutation of integers in some interval

  ```ruby
  Torch.randperm(3) # tensor([2, 0, 1])
  ```

- `zeros` returns a tensor filled with all zeros

  ```ruby
  Torch.zeros(3) # tensor([0, 0, 0])
  ```

## LibTorch Installation

[Download LibTorch](https://pytorch.org/) and run:

```sh
bundle config build.torch-rb --with-torch-dir=/path/to/libtorch
```

### Homebrew

For Mac, you can use Homebrew.

```sh
brew install ankane/brew/libtorch
```

Then install the gem (no need for `--with-torch-dir`).

## rbenv

This library uses [Rice](https://github.com/jasonroelofs/rice) to interface with LibTorch. Rice and earlier versions of rbenv don’t play nicely together. If you encounter an error during installation, upgrade ruby-build and reinstall your Ruby version.

```sh
brew upgrade ruby-build
rbenv install [version]
```

## History

View the [changelog](https://github.com/ankane/torch-rb/blob/master/CHANGELOG.md)

## Contributing

Everyone is encouraged to help improve this project. Here are a few ways you can help:

- [Report bugs](https://github.com/ankane/torch-rb/issues)
- Fix bugs and [submit pull requests](https://github.com/ankane/torch-rb/pulls)
- Write, clarify, or fix documentation
- Suggest or add new features

To get started with development:

```sh
git clone https://github.com/ankane/torch-rb.git
cd torch
bundle install
bundle exec rake compile
bundle exec rake test
```

Here are some good resources for contributors:

- [PyTorch API](https://pytorch.org/docs/stable/torch.html)
- [PyTorch C++ API](https://pytorch.org/cppdocs/)
- [Tensor Creation API](https://pytorch.org/cppdocs/notes/tensor_creation.html)
- [Using the PyTorch C++ Frontend](https://pytorch.org/tutorials/advanced/cpp_frontend.html)
