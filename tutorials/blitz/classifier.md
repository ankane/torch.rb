# Training a Classifier

This is it. You have seen how to define neural networks, compute loss and make updates to the weights of the network.

Now you might be thinking,

## What about data?

Generally, when you have to deal with image, text, audio or video data, you can use standard Ruby gems that load data into an array. Then you can convert this array into a `Torch::Tensor`.

Specifically for vision, we have created a gem called `torchvision`, that has data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz., `TorchVision::Datasets` and `Torch::Utils::Data::DataLoader`.

This provides a huge convenience and avoids writing boilerplate code.

For this tutorial, we will use the CIFAR10 dataset. It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

## Training an image classifier

We will do the following steps in order:

1. Load and normalize the CIFAR10 training and test datasets using `torchvision`
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

### 1. Load and normalize CIFAR10

Using `torchvision`, it’s extremely easy to load CIFAR10.

```ruby
require "torch"
require "torchvision"
```

The output of torchvision datasets are vips images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].

```ruby
transform = TorchVision::Transforms::Compose.new([
  TorchVision::Transforms::ToTensor.new,
  TorchVision::Transforms::Normalize.new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

batch_size = 4

trainset = TorchVision::Datasets::CIFAR10.new("./data", train: true, download: true, transform: transform)
trainloader = Torch::Utils::Data::DataLoader.new(trainset, batch_size: batch_size, shuffle: true)

testset = TorchVision::Datasets::CIFAR10.new("./data", train: false, download: true, transform: transform)
testloader = Torch::Utils::Data::DataLoader.new(testset, batch_size: batch_size, shuffle: false)

classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
```

Out:

```text
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz...
Files already downloaded and verified
```

Let us show some of the training images, for fun.

```ruby
require "matplotlib/iruby"

Matplotlib::IRuby.activate

# method to show an image
def imshow(img)
  img = img / 2 + 0.5 # unnormalize
  plt = Matplotlib::Pyplot
  plt.imshow(img.permute([1, 2, 0]).to_a)
  plt.show
end

# get some random training images
images, labels = trainloader.first

# show images
imshow(TorchVision::Utils.make_grid(images))
# print labels
puts labels.to_a.map { |label| classes[label] }.join(" ")
```

Out:

```text
car bird ship car
```

### 2. Define a Convolutional Neural Network

Copy the neural network from the Neural Networks section before and modify it to take 3-channel images (instead of 1-channel images as it was defined).

```ruby
class MyNet < Torch::NN::Module
  def initialize
    super()
    @conv1 = Torch::NN::Conv2d.new(3, 6, 5)
    @pool = Torch::NN::MaxPool2d.new(2, stride: 2)
    @conv2 = Torch::NN::Conv2d.new(6, 16, 5)
    @fc1 = Torch::NN::Linear.new(16 * 5 * 5, 120)
    @fc2 = Torch::NN::Linear.new(120, 84)
    @fc3 = Torch::NN::Linear.new(84, 10)
  end

  def forward(x)
    x = @pool.call(Torch::NN::F.relu(@conv1.call(x)))
    x = @pool.call(Torch::NN::F.relu(@conv2.call(x)))
    x = Torch.flatten(x, 1) # flatten all dimensions except batch
    x = Torch::NN::F.relu(@fc1.call(x))
    x = Torch::NN::F.relu(@fc2.call(x))
    @fc3.call(x)
  end
end

net = MyNet.new
```

### 3. Define a Loss function and optimizer

Let’s use a Classification Cross-Entropy loss and SGD with momentum.

```ruby
criterion = Torch::NN::CrossEntropyLoss.new
optimizer = Torch::Optim::SGD.new(net.parameters, lr: 0.001, momentum: 0.9)
```

### 4. Train the network

This is when things start to get interesting. We simply have to loop over our data iterator, and feed the inputs to the network and optimize.

```ruby
2.times do |epoch| # loop over the dataset multiple times
  running_loss = 0.0
  trainloader.each_with_index do |data, i|
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    # zero the parameter gradients
    optimizer.zero_grad

    # forward + backward + optimize
    outputs = net.call(inputs)
    loss = criterion.call(outputs, labels)
    loss.backward
    optimizer.step

    # print statistics
    running_loss += loss.item
    if i % 2000 == 1999 # print every 2000 mini-batches
      puts "[%d, %5d] loss: %.3f" % [epoch + 1, i + 1, running_loss / 2000]
      running_loss = 0.0
    end
  end
end

puts "Finished Training"
```

Out:

```text
[1,  2000] loss: 2.278
[1,  4000] loss: 1.936
[1,  6000] loss: 1.701
[1,  8000] loss: 1.598
[1, 10000] loss: 1.512
[1, 12000] loss: 1.465
[2,  2000] loss: 1.416
[2,  4000] loss: 1.367
[2,  6000] loss: 1.348
[2,  8000] loss: 1.331
[2, 10000] loss: 1.321
[2, 12000] loss: 1.281
Finished Training
```

Let’s quickly save our trained model:

```ruby
path = "./cifar_net.pth"
Torch.save(net.state_dict, path)
```

### 5. Test the network on the test data

We have trained the network for 2 passes over the training dataset. But we need to check if the network has learnt anything at all.

We will check this by predicting the class label that the neural network outputs, and checking it against the ground-truth. If the prediction is correct, we add the sample to the list of correct predictions.

Okay, first step. Let us display an image from the test set to get familiar.

```ruby
images, labels = testloader.first

# print images
imshow(TorchVision::Utils.make_grid(images))
puts "GroundTruth: #{labels.to_a.map { |label| classes[label] }.join(" ")}"
```

Out:

```text
GroundTruth: cat ship ship plane
```

Next, let’s load back in our saved model (note: saving and re-loading the model wasn’t necessary here, we only did it to illustrate how to do so):

```ruby
net = MyNet.new
net.load_state_dict(Torch.load(path))
```

Okay, now let us see what the neural network thinks these examples above are:

```ruby
outputs = net.call(images)
```

The outputs are energies for the 10 classes. The higher the energy for a class, the more the network thinks that the image is of the particular class. So, let’s get the index of the highest energy:

```ruby
_, predicted = Torch.max(outputs, 1)

puts "Predicted: #{predicted.to_a.map { |pred| classes[pred] }.join(" ")}"
```

Out:

```text
Predicted: cat ship car plane
```

The results seem pretty good.

Let us look at how the network performs on the whole dataset.

```ruby
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
Torch.no_grad do
  testloader.each do |data|
    images, labels = data
    # calculate outputs by running images through the network
    outputs = net.call(images)
    # the class with the highest energy is what we choose as prediction
    _, predicted = Torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum.item
  end
end

puts "Accuracy of the network on the 10000 test images: %d %%" % [100 * correct / total]
```

Out:

```text
Accuracy of the network on the 10000 test images: 52 %
```

That looks way better than chance, which is 10% accuracy (randomly picking a class out of 10 classes). Seems like the network learnt something.

Hmmm, what are the classes that performed well, and the classes that did not perform well:

```ruby
# prepare to count predictions for each class
correct_pred = classes.map { |classname| [classname, 0] }.to_h
total_pred = classes.map { |classname| [classname, 0] }.to_h

# again no gradients needed
Torch.no_grad do
  testloader.each do |data|
    images, labels = data
    outputs = net.call(images)
    _, predictions = Torch.max(outputs, 1)
    # collect the correct predictions for each class
    labels.to_a.zip(predictions.to_a) do |label, prediction|
      correct_pred[classes[label]] += 1 if label == prediction
      total_pred[classes[label]] += 1
    end
  end
end

# print accuracy for each class
correct_pred.each do |classname, correct_count|
  accuracy = 100.0 * correct_count / total_pred[classname]
  puts "Accuracy for class %5s is: %.1f %%" % [classname, accuracy]
end
```

Out:

```text
Accuracy for class plane is: 65.4 %
Accuracy for class   car is: 75.3 %
Accuracy for class  bird is: 66.3 %
Accuracy for class   cat is: 25.9 %
Accuracy for class  deer is: 20.9 %
Accuracy for class   dog is: 41.4 %
Accuracy for class  frog is: 67.9 %
Accuracy for class horse is: 43.3 %
Accuracy for class  ship is: 71.7 %
Accuracy for class truck is: 46.8 %
```

Okay, so what next?

How do we run these neural networks on the GPU?

## Training on GPU

Just like how you transfer a Tensor onto the GPU, you transfer the neural net onto the GPU.

Let’s first define our device as the first visible cuda device if we have CUDA available:

```ruby
device = Torch.device(Torch::CUDA.available? ? "cuda:0" : "cpu")
```

The rest of this section assumes that `device` is a CUDA device.

Then these methods will recursively go over all modules and convert their parameters and buffers to CUDA tensors:

```ruby
net.to(device)
```

Remember that you will have to send the inputs and targets at every step to the GPU too:

```ruby
inputs, labels = data[0].to(device), data[1].to(device)
```

Why don’t I notice MASSIVE speedup compared to CPU? Because your network is really small.

**Exercise:** Try increasing the width of your network (argument 2 of the first `Torch::NN::Conv2d`, and argument 1 of the second `Torch::NN::Conv2d` – they need to be the same number), see what kind of speedup you get.

**Goals achieved:**

- Understanding Torch.rb’s Tensor library and neural networks at a high level.
- Train a small neural network to classify images

## Where do I go next?

- [More examples](/README.md#examples)
- [More tutorials](/README.md#tutorials)
