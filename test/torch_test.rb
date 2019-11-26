require_relative "test_helper"

class TorchTest < Minitest::Test
  def test_tutorial
    x = Torch.empty(5, 3)

    x = Torch.rand(5, 3)

    x = Torch.zeros(5, 3, dtype: :long)
    assert_equal :int64, x.dtype

    x = Torch.tensor([5.5, 3])

    x = x.new_ones(5, 3, dtype: :double)
    assert_equal :float64, x.dtype

    x = Torch.randn_like(x, dtype: :float)
    assert_equal :float32, x.dtype

    assert_equal [5, 3], x.size

    y = Torch.rand(5, 3)
    x + y

    Torch.add(x, y)

    result = Torch.empty(5, 3)
    Torch.add(x, y, out: result)

    y.add!(x)

    x[0]

    x = Torch.randn(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)
    assert_equal [4, 4], x.size
    assert_equal [16], y.size
    assert_equal [2, 8], z.size

    assert_equal 16, x.numel

    x = Torch.randn(1)
    x.item

    a = Torch.ones(5)
    b = a.numo
    assert_kind_of Numo::SFloat, b
  end
end
