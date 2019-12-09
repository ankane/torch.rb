require_relative "../test_helper"

class PaddingLayersTest < Minitest::Test
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
