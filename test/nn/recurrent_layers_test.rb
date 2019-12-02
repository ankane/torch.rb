require_relative "../test_helper"

class RecurrentLayersTest < Minitest::Test
  def test_rnn
    rnn = Torch::NN::RNN.new(10, 20, num_layers: 2)
    input = Torch.randn(5, 3, 10)
    h0 = Torch.randn(2, 3, 20)

    skip "Not implemented yet"

    output, hn = rnn.call(input, hx: h0)
  end
end
