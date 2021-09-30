require_relative "test_helper"

class TensorMethodsTest < Minitest::Test
  def test_to_i
    x = Torch.tensor([1.5])
    assert_equal 1, x.to_i
  end

  def test_to_i_multiple
    error = assert_raises(Torch::Error) do
      Torch.tensor([1.5, 2.5]).to_i
    end
    assert_equal "only one element tensors can be converted to Ruby scalars", error.message
  end

  def test_to_f
    x = Torch.tensor([1.5])
    assert_equal 1.5, x.to_f
  end

  def test_to_f_multiple
    error = assert_raises(Torch::Error) do
      Torch.tensor([1.5, 2.5]).to_f
    end
    assert_equal "only one element tensors can be converted to Ruby scalars", error.message
  end

  def test_to_dtype
    x = Torch.tensor([1, 2, 3])
    assert_equal :float64, x.to(dtype: :float64).dtype
  end

  def test_to_symbol
    x = Torch.tensor([1, 2, 3])
    assert_equal :float64, x.to(:float64).dtype
  end

  def test_each
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = []
    x.each do |v|
      result << v.to_a
    end
    assert_equal [[1, 2, 3], [4, 5, 6]], result
  end

  def test_each_with_index
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = []
    indexes = []
    x.each_with_index do |v, i|
      result << v.to_a
      indexes << i
    end
    assert_equal [[1, 2, 3], [4, 5, 6]], result
    assert_equal [0, 1], indexes
  end

  def test_each_with_index_enumerable
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = []
    indexes = []
    x.each.with_index do |v, i|
      result << v.to_a
      indexes << i
    end
    assert_equal [[1, 2, 3], [4, 5, 6]], result
    assert_equal [0, 1], indexes
  end

  def test_map
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert_equal [[1, 2, 3], [4, 5, 6]], x.map { |v| v.to_a }
  end

  def test_boolean_operators
    x = Torch.tensor([true, true, false, false])
    y = Torch.tensor([true, false, true, false])
    assert_equal [true, false, false, false], (x & y).to_a
    assert_equal [true, true, true, false], (x | y).to_a
    assert_equal [false, true, true, false], (x ^ y).to_a
  end

  def test_type
    x = Torch.tensor([1, 2, 3])
    assert_equal :float64, x.type(:float64).dtype
    assert_equal :float64, x.double.dtype
    assert_equal :int32, x.int.dtype
    assert_equal :int64, x.dtype
  end

  def test_type_invalid
    error = assert_raises(Torch::Error) do
      Torch.tensor([1, 2, 3]).type(:bad)
    end
    assert_equal "Invalid type: bad", error.message
  end

  def test_type_class
    x = Torch.tensor([1, 2, 3])
    assert_equal :int64, x.dtype
    assert_equal :float32, x.type(Torch::FloatTensor).dtype
    assert_equal :int64, x.dtype
  end

  def test_type_class_invalid
    error = assert_raises(Torch::Error) do
      Torch.tensor([1, 2, 3]).type(Object)
    end
    assert_equal "Invalid type: Object", error.message
  end

  def test_zip
    x = Torch.tensor([1, 2, 3])
    y = Torch.tensor([4, 5, 6])
    expected = [
      [Torch.tensor(1), Torch.tensor(4)],
      [Torch.tensor(2), Torch.tensor(5)],
      [Torch.tensor(3), Torch.tensor(6)]
    ]
    assert_equal expected.inspect, x.zip(y).inspect
  end
end
