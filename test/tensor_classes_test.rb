require_relative "test_helper"

class TensorClassesTest < Minitest::Test
  def test_tensor
    x = Torch::Tensor.new([1, 2])
    assert_equal :float32, x.dtype
  end

  def test_float_tensor
    x = Torch::FloatTensor.new([1, 2])
    assert_equal :float32, x.dtype
  end

  def test_long_tensor
    x = Torch::LongTensor.new([1, 2])
    assert_equal :int64, x.dtype
  end
end
