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
