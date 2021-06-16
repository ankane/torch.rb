require_relative "test_helper"

class SpecialTest < Minitest::Test
  def test_entr
    a = Torch.arange(-0.5, 1, 0.5)
    assert_elements_in_delta [-Float::INFINITY, 0, 0.3466], Torch::Special.entr(a).to_a
  end

  def test_erf
    assert_elements_in_delta [0, -0.8427, 1], Torch::Special.erf(Torch.tensor([0, -1.0, 10])).to_a
  end

  def test_erfc
    assert_elements_in_delta [1, 1.8427, 0], Torch::Special.erfc(Torch.tensor([0, -1.0, 10])).to_a
  end

  def test_erfinv
    assert_elements_in_delta [0, 0.4769, -Float::INFINITY], Torch::Special.erfinv(Torch.tensor([0, 0.5, -1])).to_a
  end

  def test_expit
    t = Torch.tensor([1.0, 2, 3])
    assert_elements_in_delta [0.7311, 0.8808, 0.9526], Torch::Special.expit(t).to_a
  end
end
