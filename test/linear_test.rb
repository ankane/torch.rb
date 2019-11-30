require_relative "test_helper"

class LinearTest < Minitest::Test
  def test_bilinear
    m = Torch::NN::Bilinear.new(20, 30, 40)
    input1 = Torch.randn(128, 20)
    input2 = Torch.randn(128, 30)
    output = m.call(input1, input2)
    assert_equal [128, 40], output.size
    assert m.inspect
  end

  def test_identity
    m = Torch::NN::Identity.new(54, unused_argument1: 0.1, unused_argument2: false)
    input = Torch.randn(128, 20)
    output = m.call(input)
    assert_equal [128, 20], output.size
    assert m.inspect
  end

  def test_linear
    m = Torch::NN::Linear.new(20, 30)
    input = Torch.randn(128, 20)
    output = m.call(input)
    assert_equal [128, 30], output.size
    assert m.inspect
  end
end
