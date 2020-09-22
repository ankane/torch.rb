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

  def test_getter
    x = Torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert_equal [0, 1, 2], x[0].to_a
    assert_equal 5, x[1, 2].item
    assert_equal [0, 1], x[0, 0..1].to_a
    assert_equal [[[0, 1, 2], [3, 4, 5]]], x[true].to_a
    assert_equal [[[0, 1, 2], [3, 4, 5]]], x[nil].to_a
    assert_equal [1, 2], x[0, 1..-1].to_a
    assert_equal [1], x[0, 1...-1].to_a
    assert_equal [0, 1], x[0, 0...-1].to_a
    assert_equal [[0, 1, 2]], x[0...-1].to_a
    assert_equal [], x[false].to_a
    if RUBY_VERSION.to_f > 2.6
      assert_equal [1, 2], x[0, eval("1..")].to_a
      assert_equal [1], x[0, eval("(1...)")].to_a
    end
    if RUBY_VERSION.to_f > 2.7
      assert_equal [0, 1], x[0, eval("..2")].to_a
      assert_equal [0, 1, 2], x[0, eval("..")].to_a
    end
  end

  def test_getter_tensor
    x = Torch.tensor([1, 2, 3])
    index = Torch.tensor([false, true, false])
    assert_equal [2], x[index].to_a
  end

  def test_getter_large_integer
    x = Torch.tensor([1, 2, 3])
    error = assert_raises(RangeError) do
      x[2**64]
    end
    assert_equal "bignum too big to convert into `long long'", error.message
  end

  def test_setter_numeric
    x = Torch.tensor([1, 2, 3])
    x[1] = 9
    assert_equal [1, 9, 3], x.to_a
  end

  def test_setter_range_end
    x = Torch.tensor([1, 2, 3])
    x[1..2] = 9
    assert_equal [1, 9, 9], x.to_a
  end

  def test_setter_range_end_negative
    x = Torch.tensor([[0, 1, 2], [3, 4, 5]])
    x[1, 1..-1] = 9
    assert_equal [[0, 1, 2], [3, 9, 9]], x.to_a
  end

  def test_setter_range_end_negative_exclude_end
    x = Torch.tensor([[0, 1, 2], [3, 4, 5]])
    x[1, 1...-1] = 9
    assert_equal [[0, 1, 2], [3, 9, 5]], x.to_a
  end

  def test_setter_tensor
    x = Torch.tensor([1, 2, 3])
    index = Torch.tensor([false, true, false])
    x[index] = 9
    assert_equal [1, 9, 3], x.to_a
  end

  def test_fill_diagonal
    x = Torch.ones(2, 2)
    assert_equal [[3, 1], [1, 3]], x.fill_diagonal!(3).to_a
    assert_equal [[3, 1], [1, 3]], x.to_a
  end

  def test_chunk
    x = Torch.ones(2, 2)
    assert_equal [[[1, 1], [1, 1]]], x.chunk(1, 1).map(&:to_a)
    assert_equal [[[1], [1]], [[1], [1]]], x.chunk(2, 1).map(&:to_a)
  end

  def test_class_chunk
    x = Torch.ones(2, 2)
    assert_equal [[[1, 1], [1, 1]]], Torch.chunk(x, 1, 1).map(&:to_a)
    assert_equal [[[1], [1]], [[1], [1]]], Torch.chunk(x, 2, 1).map(&:to_a)
  end

  def test_split
    x = Torch.ones(2, 2)
    assert_equal [[[1, 1], [1, 1]]], x.split(2, 1).map(&:to_a)
    assert_equal [[[1], [1]], [[1], [1]]], x.split(1, 1).map(&:to_a)
  end

  def test_class_split
    x = Torch.ones(2, 2)
    assert_equal [[[1, 1], [1, 1]]], Torch.split(x, 2, 1).map(&:to_a)
    assert_equal [[[1], [1]], [[1], [1]]], Torch.split(x, 1, 1).map(&:to_a)
  end

  def test_unbind
    x = Torch.tensor([[[0, 1, 2]], [[3, 4, 5]], [[6, 7, 8]]])
    assert_equal [[[0, 1, 2]], [[3, 4, 5]], [[6, 7, 8]]], x.unbind.map(&:to_a)
    assert_equal [[[0, 1, 2], [3, 4, 5], [6, 7, 8]]], x.unbind(1).map(&:to_a)
    assert_equal [[[0], [3], [6]], [[1], [4], [7]], [[2], [5], [8]]], x.unbind(2).map(&:to_a)
  end

  def test_class_unbind
    x = Torch.tensor([[[0, 1, 2]], [[3, 4, 5]], [[6, 7, 8]]])
    assert_equal [[[0, 1, 2]], [[3, 4, 5]], [[6, 7, 8]]], Torch.unbind(x).map(&:to_a)
    assert_equal [[[0, 1, 2], [3, 4, 5], [6, 7, 8]]], Torch.unbind(x, 1).map(&:to_a)
    assert_equal [[[0], [3], [6]], [[1], [4], [7]], [[2], [5], [8]]], Torch.unbind(x, 2).map(&:to_a)
  end

  def test_permute
    x = Torch.tensor([[1, 2, 3]])
    assert_equal [[1], [2], [3]], x.permute([1, 0]).to_a
    assert_equal [[1], [2], [3]], x.permute(1, 0).to_a
  end

  def test_permute_bad
    error = assert_raises(ArgumentError) do
      Torch.tensor([[1, 2, 3]]).permute(1, 0.0)
    end
    # PyTorch returns pos 2, but pos 1 might be more intuitive
    assert_equal "argument 'dims' must be array of ints, but found element of type Float at pos 2", error.message
  end

  def test_new_full
    x = Torch.tensor([[1, 2, 3]])
    assert_equal [[0, 0], [0, 0]], x.new_full([2, 2], 0).to_a
  end
end
