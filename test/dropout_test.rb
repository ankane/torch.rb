require_relative "test_helper"

class DropoutTest < Minitest::Test
  def test_dropout
    m = Torch::NN::Dropout.new(p: 0.2)
    input = Torch.randn(20, 16)
    output = m.call(input)
    assert m.inspect
  end
end
