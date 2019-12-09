require_relative "../test_helper"

class RecurrentLayersTest < Minitest::Test
  def test_rnn
    Torch.manual_seed(1)
    rnn = Torch::NN::RNN.new(10, 20, num_layers: 2)
    input = Torch.randn(5, 3, 10)
    h0 = Torch.randn(2, 3, 20)
    output, hn = rnn.call(input, hx: h0)
    assert_equal [5, 3, 20], output.size
    assert_equal [2, 3, 20], hn.size
    expected = [0.7178, 0.1506, 0.4693, 0.0831, 0.7573, 0.2792, -0.1586, 0.2291]
    assert_elements_in_delta expected, hn[0][0][0...8].to_a
  end
end
