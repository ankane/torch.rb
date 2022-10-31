require_relative "test_helper"

class SpecialTest < Minitest::Test
  def test_entr
    a = Torch.arange(-0.5, 1, 0.5)
    assert_tensor [-Float::INFINITY, 0, 0.3466], Torch::Special.entr(a)
  end

  def test_erf
    assert_tensor [0, -0.8427, 1], Torch::Special.erf(Torch.tensor([0, -1.0, 10]))
  end

  def test_erfc
    assert_tensor [1, 1.8427, 0], Torch::Special.erfc(Torch.tensor([0, -1.0, 10]))
  end

  def test_erfinv
    assert_tensor [0, 0.4769, -Float::INFINITY], Torch::Special.erfinv(Torch.tensor([0, 0.5, -1]))
  end

  def test_expit
    t = Torch.tensor([1.0, 2, 3])
    assert_tensor [0.7311, 0.8808, 0.9526], Torch::Special.expit(t)
  end

  def test_expm1
    assert_tensor [0, 1], Torch::Special.expm1(Torch.tensor([0, Math.log(2)]))
  end

  def test_expm2
    assert_tensor [1, 2, 8, 16], Torch::Special.exp2(Torch.tensor([0, Math.log2(2), 3, 4]))
  end

  def test_gammaln
    a = Torch.arange(0.5, 2, 0.5)
    assert_tensor [0.5724, 0, -0.1208], Torch::Special.gammaln(a)
  end

  def test_i0e
    assert_tensor [1, 0.4658, 0.3085, 0.2430, 0.2070], Torch::Special.i0e(Torch.arange(5, dtype: :float32))
  end

  def test_logit
    a = Torch.tensor([1, 2, 3])
    assert_tensor [13.8023, 13.8023, 13.8023], Torch::Special.logit(a, eps: 1e-6)
  end

  def test_xlog1py
    x = Torch.tensor([1, 2, 3])
    y = Torch.tensor([3, 2, 1])
    assert_tensor [1.3863, 2.1972, 2.0794], Torch::Special.xlog1py(x, y)
  end
end
