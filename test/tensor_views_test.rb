require_relative "test_helper"

class TensorViewsTest < Minitest::Test
  def test_contiguous
    base = Torch.tensor([[0, 1], [2, 3]])
    assert base.contiguous?
    t = base.transpose(0, 1)
    refute t.contiguous?
    c = t.contiguous
    assert c.contiguous?
  end
end
