require_relative "test_helper"

class ArgParserTest < Minitest::Test
  def test_works
    x = Torch.tensor([1.0, 2, -3])
    assert_equal [1, 2, 3], Torch.abs(x).to_a
  end

  def test_out_works
    x = Torch.tensor([1.0, 2, -3])
    out = Torch.empty(3)
    Torch.abs(x, out: out)
    assert_equal [1, 2, 3], out.to_a
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
end
