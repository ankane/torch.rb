require_relative "../test_helper"

class ConvolutionLayersTest < Minitest::Test
  def test_conv2d
    m = Torch::NN::Conv2d.new(16, 33, 3, stride: 2)
    m = Torch::NN::Conv2d.new(16, 33, [3, 5], stride: [2, 1], padding: [4, 2])
    m = Torch::NN::Conv2d.new(16, 33, [3, 5], stride: [2, 1], padding: [4, 2], dilation: [3, 1])
    input = Torch.randn(20, 16, 50, 100)
    output = m.call(input)
  end
end
