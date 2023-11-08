# Tensors

Tensors are a specialized data structure that are very similar to arrays and matrices. In Torch.rb, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.

Tensors are similar to Numo’s narrays, except that tensors can run on GPUs or other specialized hardware to accelerate computing. If you’re familiar with narrays, you’ll be right at home with the Tensor API. If not, follow along in this quick API walkthrough.

```ruby
require "torch"
require "numo/narray"
```

## Tensor Initialization

Tensors can be initialized in various ways. Take a look at the following examples:

**Directly from data**

Tensors can be created directly from data. The data type is automatically inferred.

```ruby
data = [[1, 2], [3, 4]]
x_data = Torch.tensor(data)
```

**From a Numo array**

Tensors can be created from Numo arrays (and vice versa - see [Numo](#numo)).

```ruby
numo_array = Numo::NArray.cast(data)
x_numo = Torch.from_numo(numo_array)
```

**From another tensor**

The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.

```ruby
x_ones = Torch.ones_like(x_data) # retains the properties of x_data
puts "Ones Tensor:\n#{x_ones}\n\n"

x_rand = Torch.rand_like(x_data, dtype: :float) # overrides the datatype of x_data
puts "Random Tensor:\n#{x_rand}"
```

Out:

```text
Ones Tensor:
tensor([[1, 1],
        [1, 1]])

Random Tensor:
tensor([[0.3336, 0.5387],
        [0.0456, 0.4981]])
```

**With random or constant values**

`shape` is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.

```ruby
shape = [2, 3]
rand_tensor = Torch.rand(shape)
ones_tensor = Torch.ones(shape)
zeros_tensor = Torch.zeros(shape)

puts "Random Tensor:\n#{rand_tensor}\n\n"
puts "Ones Tensor:\n#{ones_tensor}\n\n"
puts "Zeros Tensor:\n#{zeros_tensor}"
```

Out:

```text
Random Tensor:
tensor([[0.3852, 0.1442, 0.7359],
        [0.7114, 0.6265, 0.3385]])

Ones Tensor:
tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

## Tensor Attributes

Tensor attributes describe their shape, datatype, and the device on which they are stored.

```ruby
tensor = Torch.rand(3, 4)

puts "Shape of tensor: #{tensor.shape}"
puts "Datatype of tensor: #{tensor.dtype}"
puts "Device tensor is stored on: #{tensor.device}"
```

Out:

```text
Shape of tensor: [3, 4]
Datatype of tensor: float32
Device tensor is stored on: cpu
```

## Tensor Operations

Over 100 tensor operations, including transposing, indexing, slicing, mathematical operations, linear algebra, random sampling, and more are comprehensively described [here](https://pytorch.org/docs/stable/torch.html).

Each of them can be run on the GPU (at typically higher speeds than on a CPU).

```ruby
# We move our tensor to the GPU if available
tensor = tensor.to("cuda") if Torch::CUDA.available?
```

Try out some of the operations from the list.

**Indexing and slicing**

```ruby
tensor = Torch.ones(4, 4)
tensor[0.., 1] = 0
puts tensor
```

Out:

```text
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

**Joining tensors**

You can use `Torch.cat` to concatenate a sequence of tensors along a given dimension. See also [`Torch.stack`](https://pytorch.org/docs/stable/generated/torch.stack.html), another tensor joining op that is subtly different from `Torch.cat`.

```ruby
t1 = Torch.cat([tensor, tensor, tensor], dim: 1)
puts t1
```

Out:

```text
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```


**Multiplying tensors**

```ruby
# This computes the element-wise product
puts "tensor.mul(tensor)\n#{tensor.mul(tensor)}\n\n"
# Alternative syntax:
puts "tensor * tensor\n#{tensor * tensor}"
```

Out:

```text
tensor.mul(tensor)
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor * tensor
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

This computes the matrix multiplication between two tensors

```ruby
puts "tensor.matmul(tensor.t)\n#{tensor.matmul(tensor.t)}"
```

Out:

```text
tensor.matmul(tensor.t)
tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
```

**In-place operations**

Operations that have a `!` suffix are in-place. For example: `x.copy!(y)`, `x.t!`, will change `x`.

```ruby
puts tensor, "\n"
tensor.add!(5)
puts tensor
```

Out:

```text
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

Note: In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss of history. Hence, their use is discouraged.

## Numo

### Tensor to Numo array

```ruby
t = Torch.ones(5)
puts "t: #{t}"
n = t.numo
puts "n: #{n.inspect}"
```

Out:

```text
t: tensor([1., 1., 1., 1., 1.])
n: Numo::SFloat#shape=[5]
[1, 1, 1, 1, 1]
```

### Numo array to tensor

```ruby
n = Numo::SFloat.ones(5)
puts "n: #{n.inspect}"
t = Torch.from_numo(n)
puts "t: #{t}"
```

Out:

```text
n: Numo::SFloat#shape=[5]
[1, 1, 1, 1, 1]
t: tensor([1., 1., 1., 1., 1.])
```

[Next: A Gentle Introduction to Torch::Autograd](autograd.md)
