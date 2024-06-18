# ported from PyTorch-GAN
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py
# see LICENSE-gan-examples.txt
# paper: https://arxiv.org/abs/1511.06434

require "torch"
require "torchvision"

Dir.mkdir("images") unless Dir.exist?("images")

cuda = Torch::CUDA.available?

weights_init_normal = lambda do |m|
  classname = m.class.name
  if classname.include?("Conv")
    Torch::NN::Init.normal!(m.weight.data, mean: 0.0, std: 0.02)
  elsif classname.include?("BatchNorm2d")
    Torch::NN::Init.normal!(m.weight.data, mean: 1.0, std: 0.02)
    Torch::NN::Init.constant!(m.bias.data, 0.0)
  end
end

class Generator < Torch::NN::Module
  def initialize
    super()

    @init_size = 32.div(4)
    @l1 = Torch::NN::Sequential.new(Torch::NN::Linear.new(100, 128 * @init_size ** 2))

    @conv_blocks = Torch::NN::Sequential.new(
      Torch::NN::BatchNorm2d.new(128),
      Torch::NN::Upsample.new(scale_factor: 2),
      Torch::NN::Conv2d.new(128, 128, 3, stride: 1, padding: 1),
      Torch::NN::BatchNorm2d.new(128, eps: 0.8),
      Torch::NN::LeakyReLU.new(negative_slope: 0.2, inplace: true),
      Torch::NN::Upsample.new(scale_factor: 2),
      Torch::NN::Conv2d.new(128, 64, 3, stride: 1, padding: 1),
      Torch::NN::BatchNorm2d.new(64, eps: 0.8),
      Torch::NN::LeakyReLU.new(negative_slope: 0.2, inplace: true),
      Torch::NN::Conv2d.new(64, 1, 3, stride: 1, padding: 1),
      Torch::NN::Tanh.new
    )
  end

  def forward(z)
    out = @l1.call(z)
    out = out.view(out.shape[0], 128, @init_size, @init_size)
    img = @conv_blocks.call(out)
    img
  end
end

class Discriminator < Torch::NN::Module
  def initialize
    super()

    discriminator_block = lambda do |in_filters, out_filters, bn = true|
      block = [Torch::NN::Conv2d.new(in_filters, out_filters, 3, stride: 2, padding: 1), Torch::NN::LeakyReLU.new(negative_slope: 0.2, inplace: true), Torch::NN::Dropout2d.new(p: 0.25)]
      if bn
        block << Torch::NN::BatchNorm2d.new(out_filters, eps: 0.8)
      end
      block
    end

    @model = Torch::NN::Sequential.new(
      *discriminator_block.call(1, 16, false),
      *discriminator_block.call(16, 32),
      *discriminator_block.call(32, 64),
      *discriminator_block.call(64, 128),
    )

    # The height and width of downsampled image
    ds_size = 32.div(2 ** 4)
    @adv_layer = Torch::NN::Sequential.new(Torch::NN::Linear.new(128 * ds_size ** 2, 1), Torch::NN::Sigmoid.new)
  end

  def forward(img)
    out = @model.call(img)
    out = out.view(out.shape[0], -1)
    validity = @adv_layer.call(out)
    validity
  end
end

# Loss function
adversarial_loss = Torch::NN::BCELoss.new

# Initialize generator and discriminator
generator = Generator.new
discriminator = Discriminator.new

if cuda
  generator.cuda
  discriminator.cuda
  adversarial_loss.cuda
end

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataloader = Torch::Utils::Data::DataLoader.new(
  TorchVision::Datasets::MNIST.new(
    "./data",
    train: true,
    download: true,
    transform: TorchVision::Transforms::Compose.new(
      [TorchVision::Transforms::Resize.new(32), TorchVision::Transforms::ToTensor.new, TorchVision::Transforms::Normalize.new([0.5], [0.5])]
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
