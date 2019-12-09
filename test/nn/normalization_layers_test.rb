require_relative "../test_helper"

class NormalizationLayersTest < Minitest::Test
  def test_batch_norm1d
    m = Torch::NN::BatchNorm1d.new(100)
    m = Torch::NN::BatchNorm1d.new(100, affine: false)
    input = Torch.randn(20, 100)
    output = m.call(input)
  end

  def test_batch_norm2d
    m = Torch::NN::BatchNorm2d.new(100)
    m = Torch::NN::BatchNorm2d.new(100, affine: false)
    input = Torch.randn(20, 100, 35, 45)
    output = m.call(input)
  end

  def test_batch_norm3d
    m = Torch::NN::BatchNorm3d.new(100)
    m = Torch::NN::BatchNorm3d.new(100, affine: false)
    input = Torch.randn(20, 100, 35, 45, 10)
    output = m.call(input)
  end

  def test_instance_norm1d
    m = Torch::NN::InstanceNorm1d.new(100)
    m = Torch::NN::InstanceNorm1d.new(100, affine: true)
    input = Torch.randn(20, 100, 40)
    output = m.call(input)
  end

  def test_instance_norm2d
    m = Torch::NN::InstanceNorm2d.new(100)
    m = Torch::NN::InstanceNorm2d.new(100, affine: true)
    input = Torch.randn(20, 100, 35, 45)
    output = m.call(input)
  end

  def test_instance_norm3d
    m = Torch::NN::InstanceNorm3d.new(100)
    m = Torch::NN::InstanceNorm3d.new(100, affine: true)
    input = Torch.randn(20, 100, 35, 45, 10)
    output = m.call(input)
  end
end
