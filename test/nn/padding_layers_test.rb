require_relative "../test_helper"

class PaddingLayersTest < Minitest::Test
  def test_reflection_pad1d
    m = Torch::NN::ReflectionPad1d.new(2)
    input = Torch.arange(8, dtype: :float).reshape(1, 2, 4)
    m.call(input)

    m = Torch::NN::ReflectionPad1d.new([3, 1])
    m.call(input)
  end

  def test_reflection_pad2d
    m = Torch::NN::ReflectionPad2d.new(2)
    input = Torch.arange(9, dtype: :float).reshape(1, 1, 3, 3)
    m.call(input)

    m = Torch::NN::ReflectionPad2d.new([1, 1, 2, 0])
    m.call(input)
  end

  def test_replication_pad1d
    m = Torch::NN::ReplicationPad1d.new(2)
    input = Torch.arange(8, dtype: :float).reshape(1, 2, 4)
    m.call(input)
  end

  def test_replication_pad2d
    m = Torch::NN::ReplicationPad2d.new(2)
    input = Torch.arange(9, dtype: :float).reshape(1, 1, 3, 3)
    m.call(input)
  end

  def test_replication_pad3d
    m = Torch::NN::ReplicationPad3d.new(3)
    input = Torch.randn(16, 3, 8, 320, 480)
    output = m.call(input)

    m = Torch::NN::ReplicationPad3d.new([3, 3, 6, 6, 1, 1])
    output = m.call(input)
  end

  def test_zero_pad2d
    m = Torch::NN::ZeroPad2d.new(2)
    input = Torch.randn(1, 1, 3, 3)
    m.call(input)

    m = Torch::NN::ZeroPad2d.new([1, 1, 2, 0])
    m.call(input)
  end

  def test_constant_pad1d
    m = Torch::NN::ConstantPad1d.new(2, 3.5)
    input = Torch.randn(1, 2, 4)
    m.call(input)
  end

  def test_constant_pad2d
    m = Torch::NN::ConstantPad2d.new(2, 3.5)
    input = Torch.randn(1, 2, 2)
    m.call(input)
  end

  def test_constant_pad3d
    m = Torch::NN::ConstantPad3d.new(3, 3.5)
    input = Torch.randn(16, 3, 10, 20, 30)
    m.call(input)
  end
end
