# Transfer Learning for Computer Vision Tutorial

Ported from [Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) by [Sasank Chilamkurthy](https://chsasank.github.io/)

Available under the same license ([BSD-3-Clause](LICENSE-transfer-learning-tutorial.txt))

---

In this tutorial, you will learn how to train a convolutional neural network for image classification using transfer learning. You can read more about the transfer learning at [cs231n notes](https://cs231n.github.io/transfer-learning/).

Quoting these notes, “In practice, very few people train an entire Convolutional Network from scratch (with random initialization), because it is relatively rare to have a dataset of sufficient size. Instead, it is common to pretrain a ConvNet on a very large dataset (e.g. ImageNet, which contains 1.2 million images with 1000 categories), and then use the ConvNet either as an initialization or a fixed feature extractor for the task of interest.”

These two major transfer learning scenarios look as follows:

- **Finetuning the convnet**: Instead of random initialization, we initialize the network with a pretrained network, like the one that is trained on imagenet 1000 dataset. Rest of the training looks as usual.
- **ConvNet as fixed feature extractor**: Here, we will freeze the weights for all of the network except that of the final fully connected layer. This last fully connected layer is replaced with a new one with random weights and only this layer is trained.

```ruby
# License: BSD
# Author: Sasank Chilamkurthy

require "torch"
require "torchvision"
require "numo/narray"
require "matplotlib/iruby"

Matplotlib::IRuby.activate

plt = Matplotlib::Pyplot
plt.ion # interactive mode
```

## Load Data

We will use `TorchVision` and `Torch::Utils::Data` modules for loading the data.

The problem we're going to solve today is to train a model to classify **ants** and **bees**. We have about 120 training images each for ants and bees. There are 75 validation images for each class. Usually, this is a very small dataset to generalize upon, if trained from scratch. Since we are using transfer learning, we should be able to generalize reasonably well.

This dataset is a very small subset of imagenet.

Note: Download the data from [here](https://download.pytorch.org/tutorial/hymenoptera_data.zip) and extract it to the current directory.

```ruby
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
  "train" => TorchVision::Transforms::Compose.new([
    TorchVision::Transforms::RandomResizedCrop.new(224),
    TorchVision::Transforms::RandomHorizontalFlip.new,
    TorchVision::Transforms::ToTensor.new,
    TorchVision::Transforms::Normalize.new([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ]),
  "val" => TorchVision::Transforms::Compose.new([
    TorchVision::Transforms::Resize.new(256),
    TorchVision::Transforms::CenterCrop.new(224),
    TorchVision::Transforms::ToTensor.new,
    TorchVision::Transforms::Normalize.new([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
}

data_dir = "hymenoptera_data"
image_datasets =
  ["train", "val"].to_h do |x|
    [x, TorchVision::Datasets::ImageFolder.new(File.join(data_dir, x), transform: data_transforms[x])]
  end
dataloaders =
  ["train", "val"].to_h do |x|
    [x, Torch::Utils::Data::DataLoader.new(image_datasets[x], batch_size: 4, shuffle: true)]
  end
dataset_sizes = ["train", "val"].to_h { |x| [x, image_datasets[x].size] }
class_names = image_datasets["train"].classes

device = Torch.device(Torch::CUDA.available? ? "cuda:0" : "cpu")
```

### Visualize a few images

Let’s visualize a few training images so as to understand the data augmentations.

```ruby
def imshow(inp, title: nil)
  inp = inp.numo.transpose(1, 2, 0)
  mean = Numo::SFloat.cast([0.485, 0.456, 0.406])
  std = Numo::SFloat.cast([0.229, 0.224, 0.225])
  inp = std * inp + mean
  inp = inp.clip(0, 1)
  plt = Matplotlib::Pyplot
  plt.imshow(inp.to_a)
  plt.title(title) if title
  plt.pause(0.001) # pause a bit so that plots are updated
end

# Get a batch of training data
inputs, classes = dataloaders["train"].first

# Make a grid from batch
out = TorchVision::Utils.make_grid(inputs)

imshow(out, title: classes.to_a.map { |x| class_names[x] })
```

## Training the model

Now, let’s write a general function to train a model. Here, we will illustrate:

- Scheduling the learning rate
- Saving the best model

In the following, parameter `scheduler` is an LR scheduler object from `Torch::Optim::LRScheduler`.

```ruby
train_model = lambda do |model, criterion, optimizer, scheduler, num_epochs=25|
  since = Time.now

  best_model_wts = model.state_dict.transform_values { |v| v.data.clone }
  best_acc = 0.0

  num_epochs.times do |epoch|
    puts "Epoch #{epoch}/#{num_epochs - 1}"
    puts "-" * 10

    # Each epoch has a training and validation phase
    ["train", "val"].each do |phase|
      if phase == "train"
        model.train  # Set model to training mode
      else
        model.eval   # Set model to evaluate mode
      end

      running_loss = 0.0
      running_corrects = 0

      # Iterate over data.
      dataloaders[phase].each do |inputs, labels|
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad

        # forward
        # track history if only in train
        loss = nil
        preds = nil
        Torch.set_grad_enabled(phase == "train") do
          outputs = model.call(inputs)
          _, preds = Torch.max(outputs, 1)
          loss = criterion.call(outputs, labels)

          # backward + optimize only if in training phase
          if phase == "train"
            loss.backward
            optimizer.step
          end
        end

        # statistics
        running_loss += loss.item * inputs.size(0)
        running_corrects += Torch.sum(preds.eq(labels.data)).item
      end
      if phase == "train"
        scheduler.step
      end

      epoch_loss = running_loss / dataset_sizes[phase]
      epoch_acc = running_corrects.to_f / dataset_sizes[phase]

      puts "%s Loss: %.4f Acc: %.4f" % [phase, epoch_loss, epoch_acc]

      # deep copy the model
      if phase == "val" && epoch_acc > best_acc
        best_acc = epoch_acc
        best_model_wts = model.state_dict.transform_values { |v| v.data.clone }
      end
    end
    puts
  end

  time_elapsed = Time.now - since
  puts "Training complete in %.0fm %.0fs" % [time_elapsed.div(60), time_elapsed % 60]
  puts "Best val Acc: %4f" % best_acc

  # load best model weights
  model.load_state_dict(best_model_wts)
  model
end
```

### Visualizing the model predictions

Generic method to display predictions for a few images

```ruby
visualize_model = lambda do |model, num_images=6|
  was_training = model.training
  model.eval
  images_so_far = 0
  fig = plt.figure

  Torch.no_grad do
    dataloaders["val"].each_with_index do |(inputs, labels), i|
      inputs = inputs.to(device)
      labels = labels.to(device)

      outputs = model.call(inputs)
      _, preds = Torch.max(outputs, 1)

      inputs.size[0].times do |j|
        images_so_far += 1
        ax = plt.subplot(num_images.div(2), 2, images_so_far)
        ax.axis("off")
        ax.set_title("predicted: #{class_names[preds[j].item]}")
        imshow(inputs.cpu.data[j])

        if images_so_far == num_images
          model.train(mode: was_training)
          return
        end
      end
    end

    model.train(mode: was_training)
  end
end
```

## Finetuning the convnet

Use [this script](https://github.com/ankane/torchvision/blob/master/pretrained.py) to download the pretrained model.

```sh
pip install torchvision
python pretrained.py
```

Load a pretrained model and reset final fully connected layer.

```ruby
model_ft = TorchVision::Models::ResNet18.new
model_ft.load_state_dict(Torch.load("net.pth"))

num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = Torch::NN::Linear.new(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = Torch::NN::CrossEntropyLoss.new

# Observe that all parameters are being optimized
optimizer_ft = Torch::Optim::SGD.new(model_ft.parameters, lr: 0.001, momentum: 0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = Torch::Optim::LRScheduler::StepLR.new(optimizer_ft, step_size: 7, gamma: 0.1)
```

### Train and evaluate

It should take around 15-25 min on CPU. On GPU though, it takes less than a minute.

```ruby
model_ft = train_model.call(model_ft, criterion, optimizer_ft, exp_lr_scheduler, 25)
```

```ruby
visualize_model.call(model_ft)
```

## ConvNet as fixed feature extractor

Here, we need to freeze all the network except the final layer. We need to set `requires_grad: false` to freeze the parameters so that the gradients are not computed in `backward`.

```ruby
model_conv = TorchVision::Models::ResNet18.new
model_conv.load_state_dict(Torch.load("net.pth"))
model_conv.parameters.each do |param|
  param.requires_grad = false
end

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = Torch::NN::Linear.new(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = Torch::NN::CrossEntropyLoss.new

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = Torch::Optim::SGD.new(model_conv.fc.parameters, lr: 0.001, momentum: 0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = Torch::Optim::LRScheduler::StepLR.new(optimizer_conv, step_size: 7, gamma: 0.1)
```

### Train and evaluate

On CPU this will take about half the time compared to previous scenario. This is expected as gradients don't need to be computed for most of the network. However, forward does need to be computed.

```ruby
model_conv = train_model.call(model_conv, criterion, optimizer_conv, exp_lr_scheduler, 25)
```

```ruby
visualize_model.call(model_conv)

plt.ioff
plt.show
```
