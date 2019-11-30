require_relative "../test_helper"

class ActivationsTest < Minitest::Test
  def test_leaky_relu
    m = Torch::NN::LeakyReLU.new
    input = Torch.randn(2)
    output = m.call(input)
  end

  def test_prelu
    m = Torch::NN::PReLU.new
    input = Torch.randn(2)
    output = m.call(input)
  end

  def test_relu
    m = Torch::NN::ReLU.new
    input = Torch.randn(2)
    output = m.call(input)
  end

  def test_sigmoid
    m = Torch::NN::Sigmoid.new
    input = Torch.randn(2)
    output = m.call(input)
  end

  def test_softplus
    m = Torch::NN::Softplus.new
    input = Torch.randn(2)
    output = m.call(input)
  end
end
