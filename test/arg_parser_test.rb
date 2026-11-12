require_relative "test_helper"

class ArgParserTest < Minitest::Test
  def test_works
    x = Torch.tensor([1.0, 2, -3])
    assert_tensor [1, 2, 3], Torch.abs(x)
  end

  def test_out_works
    x = Torch.tensor([1.0, 2, -3])
    out = Torch.empty(3)
    Torch.abs(x, out: out)
    assert_tensor [1, 2, 3], out
  end

  def test_too_few_arguments
    error = assert_raises(ArgumentError) do
      Torch.abs
    end
    assert_equal "abs() missing 1 required positional arguments: \"input\"", error.message
  end

  def test_too_many_arguments
    error = assert_raises(ArgumentError) do
      Torch.abs(1, 2)
    end
    assert_equal "abs() takes 1 positional argument but 2 were given", error.message
  end

  def test_bad_keyword
    x = Torch.tensor([-1])
    error = assert_raises(ArgumentError) do
      Torch.abs(x, bad: 2)
    end
    assert_equal "abs() got an unexpected keyword argument 'bad'", error.message
  end

  def test_bad_keyword_value
    error = assert_raises(ArgumentError) do
      Torch.abs(1)
    end
    assert_equal "abs(): argument 'input' (position 1) must be Tensor, not Integer", error.message
  end

  def test_bad_keyword_value_out
    x = Torch.tensor([-1])
    error = assert_raises(ArgumentError) do
      Torch.abs(x, out: 2)
    end
    assert_equal "abs(): argument 'out' must be Tensor, not Integer", error.message
  end

  def test_nil_argument
    error = assert_raises(ArgumentError) do
      Torch.abs(nil)
    end
    assert_equal "abs(): argument 'input' (position 1) must be Tensor, not NilClass", error.message
  end

  # 0-dim tensors also match Scalar parameters, so Tensor overloads must be
  # tried first, or the result takes the default dtype (expected dtypes and
  # values from PyTorch)
  def test_zero_dim_tensors_prefer_tensor_overloads
    a = Torch.tensor(0.3, dtype: :float64)
    b = Torch.tensor(0.6, dtype: :float64)
    c = Torch.tensor(true)
    f = Torch.tensor(0.5)
    i = Torch.tensor(2)

    [
      [Torch.where(c, a, b), 0.3],
      [Torch.where(c, 1.0, b), 1.0],
      [Torch.where(c, a, 1.0), 0.3],
      [Torch.add(i, a), 2.3],
      [Torch.mul(f, a), 0.15],
      [Torch.sub(f, a), 0.2],
      [Torch.pow(f, a), 0.8122523963562356],
      [Torch.clamp(f, min: a), 0.5]
    ].each do |result, expected|
      assert_equal :float64, result.dtype
      assert_in_delta expected, result.item, 1e-12
    end
  end

  # numbers given for Tensor parameters (add, sub, mul, div...) are wrapped
  # numbers, which take part in type promotion like Scalars (expected dtypes
  # and values from PyTorch)
  def test_numbers_as_tensors
    i = Torch.tensor([1, 2])
    f = Torch.tensor([1.0, 2.0])
    d = Torch.tensor([1.0, 2.0], dtype: :float64)

    assert_tensor [3, 4], i + 2, dtype: :int64
    assert_tensor [3.5, 4.5], i + 2.5, dtype: :float32
    assert_tensor [2.0, 4.0], f * 2, dtype: :float32
    assert_tensor [0.9, 1.9], d - 0.1, dtype: :float64
    assert_tensor [0.5, 1.0], i / 2, dtype: :float32
    assert_tensor [0, 1], Torch.floor_divide(i, 2), dtype: :int64
    assert_tensor [7, 8], Torch.add(i, 2, alpha: 3), dtype: :int64
    assert_equal [Complex(2, 2), Complex(3, 2)], (f + Complex(1, 2)).to_a
    assert_equal :complex64, (f + Complex(1, 2)).dtype
  end

  def test_zero_dim_tensors_scalar_overloads
    f = Torch.tensor(0.5)
    a = Torch.tensor(0.3, dtype: :float64)
    c = Torch.tensor(true)

    result = f.masked_fill(c, a)
    assert_equal :float32, result.dtype
    assert_in_delta 0.3, result.item
  end
end
