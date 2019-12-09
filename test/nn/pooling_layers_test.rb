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
    pool = Torch::NN::MaxPool1d.new(2, stride: 2, return_indices: true)
    unpool = Torch::NN::MaxUnpool1d.new(2, stride: 2)
    input = Torch.tensor([[[1.0, 2, 3, 4, 5, 6, 7, 8]]])
    output, indices = pool.call(input)

    skip "Not implemented yet"

    unpool.call(output, indices)
  end

  def test_max_unpool2d
    pool = Torch::NN::MaxPool2d.new(2, stride: 2, return_indices: true)
    unpool = Torch::NN::MaxUnpool2d.new(2, stride: 2)
    input = Torch.tensor([[[[1.0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
    output, indices = pool.call(input)

    skip "Not implemented yet"

    unpool.call(output, indices)
  end

  def test_max_unpool3d
    pool = Torch::NN::MaxPool3d.new(3, stride: 2, return_indices: true)
    unpool = Torch::NN::MaxUnpool3d.new(3, stride: 2)
    output, indices = pool.call(Torch.randn(20, 16, 51, 33, 15))

    skip "Not implemented yet"

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

  def test_fractional_max_pool2d
    skip "Not implemented yet"

    m = Torch::NN::FractionalMaxPool2d.new(3, output_size: [13, 12])
    m = Torch::NN::FractionalMaxPool2d.new(3, output_ratio: [0.5, 0.5])
    input = Torch.randn(20, 16, 50, 32)
    output = m.call(input)
  end

  def test_lp_pool1d
    skip "Not implemented yet"

    m = Torch::NN::LPPool1d.new(2, 3, stride: 2)
    input = Torch.randn(20, 16, 50)
    output = m.call(input)
  end

  def test_lp_pool2d
    skip "Not implemented yet"

    m = Torch::NN::LPPool2d.new(2, 3, stride: 2)
    m = Torch::NN::LPPool2d.new(1.2, [3, 2], stride: [2, 1])
    input = Torch.randn(20, 16, 50, 32)
    output = m.call(input)
  end

  def test_adaptive_max_pool1d
    skip "Not implemented yet"

    m = Torch::NN::AdaptiveMaxPool1d.new(5)
    input = Torch.randn(1, 64, 8)
    output = m.call(input)
  end

  def test_adaptive_max_pool2d
    skip "Not implemented yet"

    m = Torch::NN::AdaptiveMaxPool2d.new([5, 7])
    input = Torch.randn(1, 64, 8, 9)
    output = m.call(input)

    m = Torch::NN::AdaptiveMaxPool2d.new(7)
    input = Torch.randn(1, 64, 10, 9)
    output = m.call(input)

    m = Torch::NN::AdaptiveMaxPool2d.new([nil, 7])
    input = Torch.randn(1, 64, 10, 9)
    output = m.call(input)
  end

  def test_adaptive_max_pool3d
    skip "Not implemented yet"

    m = Torch::NN::AdaptiveMaxPool3d.new([5, 7, 9])
    input = Torch.randn(1, 64, 8, 9, 10)
    output = m.call(input)

    m = Torch::NN::AdaptiveMaxPool3d.new(7)
    input = Torch.randn(1, 64, 10, 9, 8)
    output = m.call(input)

    m = Torch::NN::AdaptiveMaxPool3d.new([7, nil, nil])
    input = Torch.randn(1, 64, 10, 9, 8)
    output = m.call(input)
  end

  def test_adaptive_avg_pool1d
    skip "Not implemented yet"

    m = Torch::NN::AdaptiveAvgPool1d.new(5)
    input = Torch.randn(1, 64, 8)
    output = m.call(input)
  end

  def test_adaptive_avg_pool2d
    skip "Not implemented yet"

    m = Torch::NN::AdaptiveAvgPool2d.new([5, 7])
    input = Torch.randn(1, 64, 8, 9)
    output = m.call(input)

    m = Torch::NN::AdaptiveAvgPool2d.new(7)
    input = Torch.randn(1, 64, 10, 9)
    output = m.call(input)

    m = Torch::NN::AdaptiveAvgPool2d.new([nil, 7])
    input = Torch.randn(1, 64, 10, 9)
    output = m.call(input)
  end

  def test_adaptive_avg_pool3d
    skip "Not implemented yet"

    m = Torch::NN::AdaptiveAvgPool3d.new([5, 7, 9])
    input = Torch.randn(1, 64, 8, 9, 10)
    output = m.call(input)

    m = Torch::NN::AdaptiveAvgPool3d.new(7)
    input = Torch.randn(1, 64, 10, 9, 8)
    output = m.call(input)

    m = Torch::NN::AdaptiveMaxPool3d.new([7, nil, nil])
    input = Torch.randn(1, 64, 10, 9, 8)
    output = m.call(input)
  end
end
