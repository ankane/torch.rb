# Neural Networks

Neural networks can be constructed using the `Torch::NN` module.

Now that you had a glimpse of `Torch::Autograd`, `Torch::NN` depends on `Torch::Autograd` to define models and differentiate them.

A `Torch::NN::Module` contains layers, and a method `forward(input)` that returns the `output`.

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule: `weight = weight - learning_rate * gradient`

## Define the network

Let’s define this network:

```ruby
require "torch"

class MyNet < Torch::NN::Module
  def initialize
    super
    # 1 input image channel, 6 output channels, 5x5 square convolution
    # kernel
    @conv1 = Torch::NN::Conv2d.new(1, 6, 5)
    @conv2 = Torch::NN::Conv2d.new(6, 16, 5)
    # an affine operation: y = Wx + b
    @fc1 = Torch::NN::Linear.new(16 * 5 * 5, 120)
    @fc2 = Torch::NN::Linear.new(120, 84)
    @fc3 = Torch::NN::Linear.new(84, 10)
  end

  def forward(x)
    # Max pooling over a (2, 2) window
    x = Torch::NN::F.max_pool2d(Torch::NN::F.relu(@conv1.call(x)), [2, 2])
    # If the size is a square, you can specify with a single number
    x = Torch::NN::F.max_pool2d(Torch::NN::F.relu(@conv2.call(x)), 2)
    x = Torch.flatten(x, 1) # flatten all dimensions except the batch dimension
    x = Torch::NN::F.relu(@fc1.call(x))
    x = Torch::NN::F.relu(@fc2.call(x))
    @fc3.call(x)
  end
end

net = MyNet.new
p net
```

Out:

```text
MyNet(
  (conv1): Conv2d(1, 6, kernel_size: [5, 5], stride: [1, 1])
  (conv2): Conv2d(6, 16, kernel_size: [5, 5], stride: [1, 1])
  (fc1): Linear(in_features: 400, out_features: 120, bias: true)
  (fc2): Linear(in_features: 120, out_features: 84, bias: true)
  (fc3): Linear(in_features: 84, out_features: 10, bias: true)
)
```

You just have to define the `forward` method, and the `backward` method (where gradients are computed) is automatically defined for you using `Torch::Autograd`. You can use any of the Tensor operations in the `forward` method.

The learnable parameters of a model are returned by `net.parameters`.

```ruby
params = net.parameters
p params.length
p params[0].size # conv1's .weight
```

Out:

```text
10
[6, 1, 5, 5]
```

Let’s try a random 32x32 input. Note: expected input size of this net (LeNet) is 32x32. To use this net on the MNIST dataset, please resize the images from the dataset to 32x32.

```ruby
input = Torch.randn(1, 1, 32, 32)
out = net.call(input)
p out
```

Out:

```text
tensor([[ 0.0372, -0.1058,  0.1373,  0.0924, -0.1012, -0.0540,  0.0306,  0.0649,
         -0.0865,  0.0951]], requires_grad: true)
```

Zero the gradient buffers of all parameters and backprops with random gradients:

```ruby
net.zero_grad
out.backward(Torch.randn(1, 10))
```

---

Note: `Torch::NN` only supports mini-batches. The entire `Torch::NN` module only supports inputs that are a mini-batch of samples, and not a single sample.

For example, `Torch::NN::Conv2d` will take in a 4D Tensor of `nSamples x nChannels x Height x Width`.

If you have a single sample, just use `input.unsqueeze(0)` to add a fake batch dimension.

---

Before proceeding further, let’s recap all the classes you’ve seen so far.

**Recap:**

- `Torch::Tensor` - A *multi-dimensional array* with support for autograd operations like `backward`. Also *holds the gradient* w.r.t. the tensor.
- `Torch::NN::Module` - Neural network module. *Convenient way of encapsulating parameters*, with helpers for moving them to GPU, exporting, loading, etc.
- `Torch::NN::Parameter` - A kind of Tensor, that is *automatically registered as a parameter when assigned as an attribute to a* `Torch:NN::Module`.

**At this point, we covered:**

- Defining a neural network
- Processing inputs and calling backward

**Still Left:**

- Computing the loss
- Updating the weights of the network

## Loss Function

A loss function takes the (output, target) pair of inputs, and computes a value that estimates how far away the output is from the target.

There are several different [loss functions](https://pytorch.org/docs/nn.html#loss-functions) under the `Torch::NN` module. A simple loss is: `Torch::NN::MSELoss` which computes the mean-squared error between the input and the target.

For example:

```ruby
output = net.call(input)
target = Torch.randn(10) # a dummy target, for example
target = target.view(1, -1) # make it the same shape as output
criterion = Torch::NN::MSELoss.new

loss = criterion.call(output, target)
p loss
```

Out:

```text
tensor(1.4371, requires_grad: true)
```

Now, if you follow `loss` in the backward direction, you will see a graph of computations that looks like this:

```text
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> flatten -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```

So, when we call `loss.backward`, the whole graph is differentiated w.r.t. the neural net parameters, and all Tensors in the graph that have `requires_grad: true` will have their `.grad` Tensor accumulated with the gradient.

## Backprop

To backpropagate the error all we have to do is to `loss.backward`. You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.

Now we shall call `loss.backward`, and have a look at conv1’s bias gradients before and after the backward.

```ruby
net.zero_grad # zeroes the gradient buffers of all parameters

puts "conv1.bias.grad before backward"
p net.conv1.bias.grad

loss.backward

puts "conv1.bias.grad after backward"
p net.conv1.bias.grad
```

Out:

```ruby
conv1.bias.grad before backward
nil
conv1.bias.grad after backward
tensor([ 0.0044, -0.0015, -0.0084,  0.0121, -0.0160,  0.0089])
```

Now, we have seen how to use loss functions.

**The only thing left to learn is:**

- Updating the weights of the network

## Update the weights

The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):

```text
weight = weight - learning_rate * gradient
```

We can implement this using simple Ruby code:

```ruby
learning_rate = 0.01
net.parameters.each do |f|
  f.data.sub!(f.grad.data * learning_rate)
end
```

However, as you use neural networks, you want to use various different update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc. To enable this, we built a small module: `Torch::Optim` that implements all these methods. Using it is very simple:

```ruby
# create your optimizer
optimizer = Torch::Optim::SGD.new(net.parameters, lr: 0.01)

# in your training loop:
optimizer.zero_grad # zero the gradient buffers
output = net.call(input)
loss = criterion.call(output, target)
loss.backward
optimizer.step # do the update
```

Note: Observe how gradient buffers had to be manually set to zero using `optimizer.zero_grad`. This is because gradients are accumulated as explained in the [Backprop](#backprop) section.

[Next: Training a Classifier](classifier.md)
