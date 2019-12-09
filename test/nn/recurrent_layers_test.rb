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

    expected = [0.0610, 0.1941, 0.0068, 0.4376, -0.5976, 0.4678, 0.4937, -0.2538]
    assert_elements_in_delta expected, output[0][0][0...8].to_a

    expected = [0.7178, 0.1506, 0.4693, 0.0831, 0.7573, 0.2792, -0.1586, 0.2291]
    assert_elements_in_delta expected, hn[0][0][0...8].to_a
  end

  def test_lstm
    Torch.manual_seed(1)

    rnn = Torch::NN::LSTM.new(10, 20, num_layers: 2)
    input = Torch.randn(5, 3, 10)
    h0 = Torch.randn(2, 3, 20)
    c0 = Torch.randn(2, 3, 20)

    output, (hn, cn) = rnn.call(input, hx: [h0, c0])

    assert_equal [5, 3, 20], output.size
    assert_equal [2, 3, 20], hn.size
    assert_equal [2, 3, 20], cn.size

    expected = [-1.6727e-01, -3.2626e-02, 2.3335e-01, 9.9146e-02, 1.7294e-01, -1.0035e-01, 2.3411e-04, 3.5714e-01]
    assert_elements_in_delta expected, output[0][0][0...8].to_a

    expected = [0.0331, 0.0044, -0.0570, -0.0453, -0.0256, 0.0393, -0.1869, -0.2357]
    assert_elements_in_delta expected, hn[0][0][0...8].to_a

    expected = [0.0565, 0.0103, -0.1633, -0.0869, -0.0622, 0.0728, -0.2707, -0.5737]
    assert_elements_in_delta expected, cn[0][0][0...8].to_a
  end
end
