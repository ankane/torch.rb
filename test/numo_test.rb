require_relative "test_helper"

class NumoTest < Minitest::Test
  def test_numo
    x = Torch.ones(2, 3)
    assert x.numo.is_a?(Numo::SFloat)

    x = Torch.ones(2, 3, dtype: :long)
    assert x.numo.is_a?(Numo::Int64)
  end

  def test_from_numo
    input = Numo::DFloat.new(2, 3).seq
    x = Torch.from_numo(input)
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
end
