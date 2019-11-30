require_relative "test_helper"

class ActivationsTest < Minitest::Test
  def test_sigmoid
    m = Torch::NN::Sigmoid.new
    input = Torch.randn(2)
    output = m.call(input)
  end
end
