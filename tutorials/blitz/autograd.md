# A Gentle Introduction to `Torch::Autograd`

`Torch::Autograd` is Torch.rb’s automatic differentiation engine that powers neural network training. In this section, you will get a conceptual understanding of how autograd helps a neural network train.

## Background

Neural networks (NNs) are a collection of nested functions that are executed on some input data. These functions are defined by *parameters* (consisting of weights and biases), which in Torch.rb are stored in
tensors.

Training a NN happens in two steps:

**Forward Propagation**: In forward prop, the NN makes its best guess about the correct output. It runs the input data through each of its functions to make this guess.

**Backward Propagation**: In backprop, the NN adjusts its parameters proportionate to the error in its guess. It does this by traversing backwards from the output, collecting the derivatives of the error with respect to the parameters of the functions (*gradients*), and optimizing the parameters using gradient descent. For a more detailed walkthrough of backprop, check out this [video from 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8>).

## Usage in PyTorch

Let’s take a look at a single training step. For this example, we load a pretrained ResNet18 model from `torchvision`. We create a random data tensor to represent a single image with 3 channels, and height & width of 64, and its corresponding `label` initialized to some random values.

```ruby
require "torch"
require "torchvision"

model = TorchVision::Models::ResNet18.new
data = Torch.rand(1, 3, 64, 64)
labels = Torch.rand(1, 1000)
```

Next, we run the input data through the model through each of its layers to make a prediction. This is the **forward pass**.

```ruby
prediction = model.call(data) # forward pass
```

We use the model’s prediction and the corresponding label to calculate the error (`loss`). The next step is to backpropagate this error through the network. Backward propagation is kicked off when we call `.backward` on the error tensor. Autograd then calculates and stores the gradients for each model parameter in the parameter’s `.grad` attribute.

```ruby
loss = (prediction - labels).sum
loss.backward # backward pass
```

Next, we load an optimizer, in this case SGD with a learning rate of 0.01 and momentum of 0.9. We register all the parameters of the model in the optimizer.

```ruby
optim = Torch::Optim::SGD.new(model.parameters, lr: 1e-2, momentum: 0.9)
```

Finally, we call `.step` to initiate gradient descent. The optimizer adjusts each parameter by its gradient stored in `.grad`.

```ruby
optim.step # gradient descent
```

At this point, you have everything you need to train your neural network.

[Next: Neural Networks](neural_networks.md)
