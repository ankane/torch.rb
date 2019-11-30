require_relative "../test_helper"

class DropoutLayersTest < Minitest::Test
  def test_dropout
    m = Torch::NN::Dropout.new(p: 0.2)
    input = Torch.randn(20, 16)
    output = m.call(input)
    assert m.inspect
  end

  def test_dropout2d
    m = Torch::NN::Dropout2d.new(p: 0.2)
    input = Torch.randn(20, 16, 32, 32)
    output = m.call(input)
  end

  def test_dropout3d
    m = Torch::NN::Dropout3d.new(p: 0.2)
    input = Torch.randn(20, 16, 4, 32, 32)
    output = m.call(input)
  end

  def test_alpha_dropout
    m = Torch::NN::AlphaDropout.new(p: 0.2)
    input = Torch.randn(20, 16)
    output = m.call(input)
  end
end
