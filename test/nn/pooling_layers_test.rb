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

  def test_max_unpool1d
    skip "Not implemented yet"

    pool = Torch::NN::MaxPool1d.new(2, stride: 2, return_indices: true)
    unpool = Torch::NN::MaxUnpool1d.new(2, stride: 2)
    input = Torch.tensor([[[1.0, 2, 3, 4, 5, 6, 7, 8]]])
    output, indices = pool.call(input)
    unpool.call(output, indices)
  end

  def test_max_unpool2d
    skip "Not implemented yet"

    pool = Torch::NN::MaxPool2d.new(2, stride: 2, return_indices: true)
    unpool = Torch::NN::MaxUnpool2d.new(2, stride: 2)
    input = Torch.tensor([[[[1.0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
    output, indices = pool.call(input)
    unpool.call(output, indices)
  end

  def test_max_unpool3d
    skip "Not implemented yet"

    pool = Torch::NN::MaxPool3d.new(3, stride: 2, return_indices: true)
    unpool = Torch::NN::MaxUnpool3d.new(3, stride: 2)
    output, indices = pool.call(Torch.randn(20, 16, 51, 33, 15))
    unpooled_output = unpool.call(output, indices)

    assert_equal [20, 16, 51, 33, 15], unpooled_output.size
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
