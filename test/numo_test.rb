require_relative "test_helper"

class NumoTest < Minitest::Test
  def test_numo
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]], dtype: :float32)
    assert x.numo.is_a?(Numo::SFloat)
    assert_equal x.to_a, x.numo.to_a

    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert x.numo.is_a?(Numo::Int64)
    assert_equal x.to_a, x.numo.to_a
  end

  def test_from_numo
    input = Numo::DFloat.new(2, 3).seq
    x = Torch.from_numo(input)

    # garbage collect to ensure underlying data is safe
    GC.start rescue nil

    assert_equal [[0, 1, 2], [3, 4, 5]], x.to_a
    assert_equal :float64, x.dtype

    input = Numo::SFloat.new(2, 3).seq
    assert_equal :float32, Torch.from_numo(input).dtype

    input = Numo::Int64.new(2, 3).seq
    assert_equal :int64, Torch.from_numo(input).dtype

    input = Numo::UInt64.new(2, 3).seq
    error = assert_raises(Torch::Error) do
      Torch.from_numo(input)
    end
    assert_equal error.message, "Cannot convert Numo::UInt64 to tensor"
  end

  def test_bridge_to
    a = Torch.ones(5)
    b = a.numo
    a.add!(1)
    assert_equal [2, 2, 2, 2, 2], a.to_a
    assert_equal [1, 1, 1, 1, 1], b.to_a
    # TODO should be
    # assert_equal [2, 2, 2, 2, 2], b.to_a
  end

  def test_bridge_from
    a = Numo::SFloat.ones(5)
    b = Torch.from_numo(a)
    a.inplace!
    a += 1
    assert_equal [2, 2, 2, 2, 2], a.to_a
    assert_equal [1, 1, 1, 1, 1], b.to_a
    # TODO should be
    # assert_equal [2, 2, 2, 2, 2], b.to_a
  end

  def test_permute
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert_equal [[1, 4], [2, 5], [3, 6]], x.permute(1, 0).to_a
    assert_equal [[1, 4], [2, 5], [3, 6]], x.permute(1, 0).numo.to_a
  end
end
