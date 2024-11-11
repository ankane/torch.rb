require_relative "../test_helper"

class ActivationsTest < Minitest::Test
  def test_elu
    m = Torch::NN::ELU.new
    input = Torch.randn(2)
    _output = m.call(input)
  end

  def test_hardshrink
    m = Torch::NN::Hardshrink.new
    input = Torch.randn(2)
    _output = m.call(input)
  end

  def test_leaky_relu
    m = Torch::NN::LeakyReLU.new
    input = Torch.randn(2)
    _output = m.call(input)
  end

  def test_log_sigmoid
    m = Torch::NN::LogSigmoid.new
    input = Torch.randn(2)
    _output = m.call(input)
  end

  def test_prelu
    m = Torch::NN::PReLU.new
    input = Torch.randn(2)
    _output = m.call(input)
  end

  def test_relu
    m = Torch::NN::ReLU.new
    input = Torch.randn(2)
    _output = m.call(input)
  end

  def test_sigmoid
    m = Torch::NN::Sigmoid.new
    input = Torch.randn(2)
    _output = m.call(input)
  end

  def test_softplus
    m = Torch::NN::Softplus.new
    input = Torch.randn(2)
    _output = m.call(input)
  end

  def test_softshrink
    m = Torch::NN::Softshrink.new
    input = Torch.randn(2)
    _output = m.call(input)
  end

  def test_softsign
    m = Torch::NN::Softsign.new
    input = Torch.randn(2)
    _output = m.call(input)
  end

  def test_tanh
    m = Torch::NN::Tanh.new
    input = Torch.randn(2)
    _output = m.call(input)
  end

  def test_tanhshrink
    m = Torch::NN::Tanhshrink.new
    input = Torch.randn(2)
    _output = m.call(input)
  end

  def test_gelu
    m = Torch::NN::GELU.new
    input = Torch.randn(2)
    _output = m.call(input)
  end
end
