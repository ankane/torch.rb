require_relative "../test_helper"

class InitTest < Minitest::Test
  def test_calculate_gain
    gain = Torch::NN::Init.calculate_gain("leaky_relu", param: 0.2)
    assert_in_delta 1.3867504905630728, gain
  end

  def test_uniform
    w = Torch.empty(3, 5)
    Torch::NN::Init.uniform!(w)
  end

  def test_normal
    w = Torch.empty(3, 5)
    Torch::NN::Init.normal!(w)
  end

  def test_constant
    w = Torch.empty(3, 5)
    Torch::NN::Init.constant!(w, 0.3)
  end

  def test_ones
    w = Torch.empty(3, 5)
    Torch::NN::Init.ones!(w)
  end

  def test_zeros
    w = Torch.empty(3, 5)
    Torch::NN::Init.zeros!(w)
  end

  def test_eye
    w = Torch.empty(3, 5)
    Torch::NN::Init.eye!(w)
  end

  def test_dirac
    w = Torch.empty(3, 16, 5, 5)
    Torch::NN::Init.dirac!(w)
  end

  def test_xavier_uniform
    w = Torch.empty(3, 5)
    Torch::NN::Init.xavier_uniform!(w, gain: Torch::NN::Init.calculate_gain("relu"))
  end

  def test_xavier_normal
    w = Torch.empty(3, 5)
    Torch::NN::Init.xavier_normal!(w)
  end

  def test_kaiming_uniform
    w = Torch.empty(3, 5)
    Torch::NN::Init.kaiming_uniform!(w, mode: "fan_in", nonlinearity: "relu")
  end

  def test_kaiming_normal
    w = Torch.empty(3, 5)
    Torch::NN::Init.kaiming_normal!(w, mode: "fan_out", nonlinearity: "relu")
  end

  def test_orthogonal
    w = Torch.empty(3, 5)
    Torch::NN::Init.orthogonal!(w)
  end

  def test_sparse
    w = Torch.empty(3, 5)
    Torch::NN::Init.sparse!(w, 0.1)
  end
end
