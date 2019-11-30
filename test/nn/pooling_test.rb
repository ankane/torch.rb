require_relative "../test_helper"

class PoolingTest < Minitest::Test
  def test_max_pool2d
    m = Torch::NN::MaxPool2d.new(3) #, stride: 2)
    m = Torch::NN::MaxPool2d.new([3, 2]) #, stride: [2, 1])
    input = Torch.randn(20, 16, 50, 32)
    output = m.call(input)
  end
end
