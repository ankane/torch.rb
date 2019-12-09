require_relative "../test_helper"

class PoolingLayersTest < Minitest::Test
  def test_max_pool1d
    m = Torch::NN::MaxPool1d.new(3, stride: 2)
    input = Torch.randn(20, 16, 50)
    output = m.call(input)
  end

  def test_max_pool2d
    m = Torch::NN::MaxPool2d.new(3, stride: 2)
    m = Torch::NN::MaxPool2d.new([3, 2], stride: [2, 1])
    input = Torch.randn(20, 16, 50, 32)
    output = m.call(input)
  end

  def test_max_pool3d
    m = Torch::NN::MaxPool3d.new(3, stride: 2)
    m = Torch::NN::MaxPool3d.new([3, 2, 2], stride: [2, 1, 2])
    input = Torch.randn(20, 16, 50,44, 31)
    output = m.call(input)
  end

  def test_avg_pool1d
    m = Torch::NN::AvgPool1d.new(3, stride: 2)
    m.call(Torch.tensor([[[1.0,2,3,4,5,6,7]]]))
  end

  def test_avg_pool2d
    m = Torch::NN::AvgPool2d.new(3, stride: 2)
    m = Torch::NN::AvgPool2d.new([3, 2], stride: [2, 1])
    input = Torch.randn(20, 16, 50, 32)
    output = m.call(input)
  end

  def test_avg_pool3d
    m = Torch::NN::AvgPool3d.new(3, stride: 2)
    m = Torch::NN::AvgPool3d.new([3, 2, 2], stride: [2, 1, 2])
    input = Torch.randn(20, 16, 50,44, 31)
    output = m.call(input)
  end
end
