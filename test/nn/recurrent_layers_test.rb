require_relative "../test_helper"

class RecurrentLayersTest < Minitest::Test
  def test_rnn
    rnn = Torch::NN::RNN.new(10, 20, num_layers: 2)
    p rnn
    input = Torch.randn(5, 3, 10)
    h0 = Torch.randn(2, 3, 20)
    output, hn = rnn.call(input, hx: h0)
  end
end
