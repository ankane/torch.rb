require_relative "../test_helper"

class ConvolutionLayersTest < Minitest::Test
  def test_conv1d
    m = Torch::NN::Conv1d.new(16, 33, 3, stride: 2)
    input = Torch.randn(20, 16, 50)
    _output = m.call(input)
  end

  def test_conv2d
    m = Torch::NN::Conv2d.new(16, 33, 3, stride: 2)
    m = Torch::NN::Conv2d.new(16, 33, [3, 5], stride: [2, 1], padding: [4, 2])
    m = Torch::NN::Conv2d.new(16, 33, [3, 5], stride: [2, 1], padding: [4, 2], dilation: [3, 1])
    input = Torch.randn(20, 16, 50, 100)
    _output = m.call(input)
  end

  def test_conv3d
    m = Torch::NN::Conv3d.new(16, 33, 3, stride: 2)
    m = Torch::NN::Conv3d.new(16, 33, [3, 5, 2], stride: [2, 1, 1], padding: [4, 2, 0])
    input = Torch.randn(20, 16, 10, 50, 100)
    _output = m.call(input)
  end

  def test_conv_transpose1d
    skip "No test"
  end

  def test_conv_transpose2d
    skip "Not implemented yet"

    m = Torch::NN::ConvTranspose2d.new(16, 33, 3, stride: 2)
    m = Torch::NN::ConvTranspose2d.new(16, 33, [3, 5], stride: [2, 1], padding: [4, 2])
    input = Torch.randn(20, 16, 50, 100)
    output = m.call(input)
    input = Torch.randn(1, 16, 12, 12)
    downsample = Torch::NN::Conv2d.new(16, 16, 3, stride: 2, padding: 1)
    upsample = Torch::NN::ConvTranspose2d.new(16, 16, 3, stride: 2, padding: 1)
    h = downsample.call(input)
    assert_equal [1, 16, 6, 6], h.size
    output = upsample.call(h, output_size: input.size())
    assert_equal [1, 16, 12, 12], output.size
  end

  def test_conv_transpose3d
    skip "Not implemented yet"

    m = Torch::NN::ConvTranspose3d.new(16, 33, 3, stride: 2)
    m = Torch::NN::ConvTranspose3d.new(16, 33, [3, 5, 2], stride: [2, 1, 1], padding: [0, 4, 2])
    input = Torch.randn(20, 16, 10, 50, 100)
    _output = m.call(input)
  end

  def test_unfold
    unfold =  Torch::NN::Unfold.new([2, 3])
    input = Torch.randn(2, 5, 3, 4)
    output = unfold.call(input)
    assert_equal [2, 30, 4], output.size
  end

  def test_fold
    fold = Torch::NN::Fold.new([4, 5], [2, 2])
    input = Torch.randn(1, 3 * 2 * 2, 12)
    output = fold.call(input)
    assert_equal [1, 3, 4, 5], output.size
  end
end
