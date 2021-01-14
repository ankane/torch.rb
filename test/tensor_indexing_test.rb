require_relative "test_helper"

class TensorIndexingTest < Minitest::Test
  def test_getter
    x = Torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert_equal [0, 1, 2], x[0].to_a
    assert_kind_of Torch::Tensor, x[0]
    assert_equal 5, x[1, 2].item
    assert_equal [0, 1], x[0, 0..1].to_a
    assert_equal [[[0, 1, 2], [3, 4, 5]]], x[true].to_a
    assert_equal [[[0, 1, 2], [3, 4, 5]]], x[nil].to_a
    assert_equal [1, 2], x[0, 1..-1].to_a
    assert_equal [1], x[0, 1...-1].to_a
    assert_equal [0, 1], x[0, 0...-1].to_a
    assert_equal [[0, 1, 2]], x[0...-1].to_a
    assert_equal [], x[false].to_a
  end

  def test_getter_endless
    skip if RUBY_VERSION.to_f < 2.6

    x = Torch.tensor([0, 1, 2])
    assert_equal [1, 2], x[eval("1..")].to_a
    assert_equal [1, 2], x[eval("(1...)")].to_a
    assert_equal [2], x[eval("-1..")].to_a
    assert_equal [2], x[eval("(-1...)")].to_a
    assert_equal [1, 2], x[eval("-2..")].to_a
    assert_equal [1, 2], x[eval("(-2...)")].to_a
  end

  def test_getter_beginless
    skip if RUBY_VERSION.to_f < 2.7

    x = Torch.tensor([0, 1, 2])
    assert_equal [0, 1], x[eval("..1")].to_a
    assert_equal [0], x[eval("...1")].to_a
    assert_equal [0, 1, 2], x[eval("..-1")].to_a
    assert_equal [0, 1], x[eval("..-2")].to_a
    assert_equal [0], x[eval("...-2")].to_a
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

  def test_setter_range_index
    x = Torch.tensor([1, 2, 3])
    x[0..1] = 0
    assert_equal [0, 0, 3], x.to_a
  end

  def test_setter_unsupported_type
    x = Torch.tensor([1, 2, 3])
    error = assert_raises(ArgumentError) do
      x[Object.new] = 1
    end
    assert_equal "Unsupported index type: Object", error.message
  end
end
