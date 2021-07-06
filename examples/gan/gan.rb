# ported from PyTorch-GAN
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
# see LICENSE-gan-examples.txt
# paper: https://arxiv.org/abs/1406.2661

require "torch"
require "torchvision"

Dir.mkdir("images") unless Dir.exist?("images")

img_shape = [1, 28, 28]

cuda = Torch::CUDA::available?

class Generator < Torch::NN::Module
  def initialize(img_shape)
    super()
    @img_shape = img_shape

    block = lambda do |in_feat, out_feat, normalize=true|
      layers = [Torch::NN::Linear.new(in_feat, out_feat)]
      if normalize
        layers << Torch::NN::BatchNorm1d.new(out_feat, eps: 0.8)
      end
      layers << Torch::NN::LeakyReLU.new(negative_slope: 0.2, inplace: true)
      layers
    end

    @model = Torch::NN::Sequential.new(
      *block.call(100, 128, false),
      *block.call(128, 256),
      *block.call(256, 512),
      *block.call(512, 1024),
      Torch::NN::Linear.new(1024, img_shape.inject(:*)),
      Torch::NN::Tanh.new
    )
  end

  def forward(z)
    img = @model.call(z)
    img = img.view(img.size(0), *@img_shape)
    img
  end
end

class Discriminator < Torch::NN::Module
  def initialize(img_shape)
    super()

    @model = Torch::NN::Sequential.new(
      Torch::NN::Linear.new(img_shape.inject(:*), 512),
      Torch::NN::LeakyReLU.new(negative_slope: 0.2, inplace: true),
      Torch::NN::Linear.new(512, 256),
      Torch::NN::LeakyReLU.new(negative_slope: 0.2, inplace: true),
      Torch::NN::Linear.new(256, 1),
      Torch::NN::Sigmoid.new,
    )
  end

  def forward(img)
    img_flat = img.view(img.size(0), -1)
    validity = @model.call(img_flat)
    validity
  end
end

# Loss function
adversarial_loss = Torch::NN::BCELoss.new

# Initialize generator and discriminator
generator = Generator.new(img_shape)
discriminator = Discriminator.new(img_shape)

if cuda
  generator.cuda
  discriminator.cuda
  adversarial_loss.cuda
end

# Configure data loader
dataloader = Torch::Utils::Data::DataLoader.new(
  TorchVision::Datasets::MNIST.new(
    "./data",
    train: true,
    download: true,
    transform: TorchVision::Transforms::Compose.new(
      [TorchVision::Transforms::Resize.new(28), TorchVision::Transforms::ToTensor.new, TorchVision::Transforms::Normalize.new([0.5], [0.5])]
    )
  ),
  batch_size: 64,
  shuffle: true
)

# Optimizers
optimizer_g = Torch::Optim::Adam.new(generator.parameters, lr: 0.0002, betas: [0.5, 0.999])
optimizer_d = Torch::Optim::Adam.new(discriminator.parameters, lr: 0.0002, betas: [0.5, 0.999])

Tensor = cuda ? Torch::CUDA::FloatTensor : Torch::FloatTensor

# ----------
#  Training
# ----------

200.times do |epoch|
  dataloader.each_with_index do |(imgs, _), i|
    # Adversarial ground truths
    valid = Tensor.new(imgs.size(0), 1).fill!(1.0)
    fake = Tensor.new(imgs.size(0), 1).fill!(0.0)

    # Configure input
    real_imgs = imgs.type(Tensor)

    # -----------------
    #  Train Generator
    # -----------------

    optimizer_g.zero_grad

    # Sample noise as generator input
    z = Tensor.new(imgs.shape[0], 100).normal!(0, 1)

    # Generate a batch of images
    gen_imgs = generator.call(z)

    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss.call(discriminator.call(gen_imgs), valid)

    g_loss.backward
    optimizer_g.step

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_d.zero_grad

    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss.call(discriminator.call(real_imgs), valid)
    fake_loss = adversarial_loss.call(discriminator.call(gen_imgs.detach), fake)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward
    optimizer_d.step

    puts "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % [epoch, 200, i, dataloader.size, d_loss.item, g_loss.item]

    batches_done = epoch * dataloader.size + i
    if batches_done % 25 == 0
      TorchVision::Utils.save_image(gen_imgs.data[0...25], "images/%d.png" % batches_done, nrow: 5, normalize: true)
    end
  end
end
