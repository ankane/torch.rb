require_relative "test_helper"

class TensorIndexingTest < Minitest::Test
  def test_getter
    x = Torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert_tensor [0, 1, 2], x[0]
    assert_equal 5, x[1, 2].item
    assert_tensor [0, 1], x[0, 0..1]
    assert_tensor [[[0, 1, 2], [3, 4, 5]]], x[true]
    assert_tensor [[[0, 1, 2], [3, 4, 5]]], x[nil]
    assert_tensor [], x[false]
    assert_tensor [[0, 1, 2]], x[0...-1]
  end

  def test_getter_range
    x = Torch.tensor([0, 1, 2])
    assert_tensor [1, 2], x[1..-1]
    assert_tensor [1], x[1...-1]
    assert_tensor [0, 1, 2], x[0..-1]
    assert_tensor [0, 1], x[0...-1]
  end

  def test_getter_endless_range
    x = Torch.tensor([0, 1, 2])
    assert_tensor [1, 2], x[1..]
    assert_tensor [1, 2], x[1...]
    assert_tensor [2], x[-1..]
    assert_tensor [2], x[-1...]
    assert_tensor [1, 2], x[-2..]
    assert_tensor [1, 2], x[-2...]
  end

  def test_getter_beginless_range
    x = Torch.tensor([0, 1, 2])
    assert_tensor [0, 1], x[..1]
    assert_tensor [0], x[...1]
    assert_tensor [0, 1, 2], x[..-1]
    assert_tensor [0, 1], x[..-2]
    assert_tensor [0], x[...-2]
  end

  def test_getter_tensor
    x = Torch.tensor([1, 2, 3])
    index = Torch.tensor([false, true, false])
    assert_tensor [2], x[index]
  end

  def test_getter_array
    x = Torch.tensor([1, 2, 3])
    index = [0, 2]
    assert_tensor [1, 3], x[index]
  end

  def test_getter_large_integer
    x = Torch.tensor([1, 2, 3])
    error = assert_raises(RangeError) do
      x[2**64]
    end
    assert_match "bignum too big to convert into", error.message
  end

  def test_getter_unsupported_type
    x = Torch.tensor([1, 2, 3])
    error = assert_raises(ArgumentError) do
      x[Object.new]
    end
    assert_equal "Unsupported index type: Object", error.message
  end

  def test_setter_numeric
    x = Torch.tensor([1, 2, 3])
    x[1] = 9
    assert_tensor [1, 9, 3], x
  end

  def test_setter_range_end
    x = Torch.tensor([1, 2, 3])
    x[1..2] = 9
    assert_tensor [1, 9, 9], x
  end

  def test_setter_range_end_negative
    x = Torch.tensor([[0, 1, 2], [3, 4, 5]])
    x[1, 1..-1] = 9
    assert_tensor [[0, 1, 2], [3, 9, 9]], x
  end

  def test_setter_range_end_negative_exclude_end
    x = Torch.tensor([[0, 1, 2], [3, 4, 5]])
    x[1, 1...-1] = 9
    assert_tensor [[0, 1, 2], [3, 9, 5]], x
  end

  def test_setter_tensor
    x = Torch.tensor([1, 2, 3])
    index = Torch.tensor([false, true, false])
    x[index] = 9
    assert_tensor [1, 9, 3], x
  end

  def test_setter_array
    x = Torch.tensor([1, 2, 3])
    x[[0, 2]] = 0
    assert_tensor [0, 2, 0], x
  end

  def test_setter_range_index
    x = Torch.tensor([1, 2, 3])
    x[0..1] = 0
    assert_tensor [0, 0, 3], x
  end

  def test_setter_unsupported_type
    x = Torch.tensor([1, 2, 3])
    error = assert_raises(ArgumentError) do
      x[Object.new] = 1
    end
    assert_equal "Unsupported index type: Object", error.message
  end
end
