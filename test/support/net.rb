class TestNet < Torch::NN::Module
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

class SimpleResidualBlock < Torch::NN::Module
  def initialize
    super()

    @relu = Torch::NN::ReLU.new

    @seq = Torch::NN::Sequential.new(
      Torch::NN::Conv2d.new(64, 128, 3, padding: 1, bias: false),
      Torch::NN::BatchNorm2d.new(128),
      @relu,
      Torch::NN::Conv2d.new(128, 128, 3, padding: 1, bias: false),
      Torch::NN::BatchNorm2d.new(128),
      @relu,
      Torch::NN::Conv2d.new(128, 64, 3, bias: false),
      Torch::NN::BatchNorm2d.new(64)
    )
  end

  def forward(x)
    @relu.forward(@seq.forward(x) + x)
  end
end
