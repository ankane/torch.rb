require_relative "../test_helper"

class ActivationsOtherTest < Minitest::Test
  def test_log_softmax
    m = Torch::NN::LogSoftmax.new
    input = Torch.randn(2, 3)
    output = m.call(input)
  end

  def test_softmax
    m = Torch::NN::Softmax.new(dim: 1)
    input = Torch.randn(2, 3)
    output = m.call(input)
  end

  def test_softmax2d
    m = Torch::NN::Softmax2d.new
    input = Torch.randn(2, 3, 12, 13)
    output = m.call(input)
  end

  def test_softmin
    m = Torch::NN::Softmin.new
    input = Torch.randn(2, 3)
    output = m.call(input)
  end
end
