require_relative "../test_helper"

class NormalizationLayersTest < Minitest::Test
  def test_batch_norm1d
    m = Torch::NN::BatchNorm1d.new(100)
    m = Torch::NN::BatchNorm1d.new(100, affine: false)
    input = Torch.randn(20, 100)
    _output = m.call(input)
  end

  def test_batch_norm2d
    m = Torch::NN::BatchNorm2d.new(100)
    m = Torch::NN::BatchNorm2d.new(100, affine: false)
    input = Torch.randn(20, 100, 35, 45)
    _output = m.call(input)
  end

  def test_batch_norm3d
    m = Torch::NN::BatchNorm3d.new(100)
    m = Torch::NN::BatchNorm3d.new(100, affine: false)
    input = Torch.randn(20, 100, 35, 45, 10)
    _output = m.call(input)
  end

  def test_group_norm
    input = Torch.randn(20, 6, 10, 10)
    m = Torch::NN::GroupNorm.new(3, 6)
    m = Torch::NN::GroupNorm.new(6, 6)
    m = Torch::NN::GroupNorm.new(1, 6)
    _output = m.call(input)
  end

  def test_sync_batch_norm
    skip "Not implemented yet"
  end

  def test_instance_norm1d
    m = Torch::NN::InstanceNorm1d.new(100)
    m = Torch::NN::InstanceNorm1d.new(100, affine: true)
    input = Torch.randn(20, 100, 40)
    _output = m.call(input)
  end

  def test_instance_norm2d
    m = Torch::NN::InstanceNorm2d.new(100)
    m = Torch::NN::InstanceNorm2d.new(100, affine: true)
    input = Torch.randn(20, 100, 35, 45)
    _output = m.call(input)
  end

  def test_instance_norm3d
    m = Torch::NN::InstanceNorm3d.new(100)
    m = Torch::NN::InstanceNorm3d.new(100, affine: true)
    input = Torch.randn(20, 100, 35, 45, 10)
    _output = m.call(input)
  end

  def test_layer_norm
    input = Torch.randn(20, 5, 10, 10)
    m = Torch::NN::LayerNorm.new(input.size[1..-1])
    m = Torch::NN::LayerNorm.new(input.size[1..-1], elementwise_affine: false)
    m = Torch::NN::LayerNorm.new([10, 10])
    m = Torch::NN::LayerNorm.new(10)
    _output = m.call(input)
  end

  def test_local_response_norm
    lrn = Torch::NN::LocalResponseNorm.new(2)
    signal_2d = Torch.randn(32, 5, 24, 24)
    signal_4d = Torch.randn(16, 5, 7, 7, 7, 7)
    _output_2d = lrn.call(signal_2d)
    _output_4d = lrn.call(signal_4d)
  end
end
