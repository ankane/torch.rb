require_relative "test_helper"

class OperationsTest < Minitest::Test
  def test_abs
    x = Torch.tensor([-1.0])
    assert_equal [1], Torch.abs(x).to_a

    out = Torch.empty(1, dtype: :float32)
    Torch.abs(x, out: out)
    assert_equal [1], out.to_a

    error = assert_raises(ArgumentError) do
      Torch.abs
    end
    assert_equal "wrong number of arguments (given 0, expected 1)", error.message
    # assert_equal "abs() missing 1 required positional arguments: \"input\"", error.message

    error = assert_raises(ArgumentError) do
      Torch.abs(1, 2)
    end
    assert_equal "wrong number of arguments (given 2, expected 1)", error.message
    # assert_equal "abs() takes 1 positional argument but 2 were given", error.message

    error = assert_raises(ArgumentError) do
      x = Torch.tensor([1])
      Torch.abs(x, bad: 2)
    end
    # assert_equal "abs() got an unexpected keyword argument 'bad'", error.message
    assert_equal "unknown keyword: bad", error.message

    error = assert_raises(ArgumentError) do
      Torch.abs(1)
    end
    assert_equal "abs(): argument 'input' must be Tensor", error.message

    error = assert_raises(ArgumentError) do
      x = Torch.tensor([1])
      Torch.abs(x, out: 2)
    end
    assert_equal "abs(): argument 'out' must be Tensor", error.message
  end

  def test_add
    x = Torch.ones(2)
    assert_equal [2, 2], (x + x).to_a
    assert_equal [2, 2], x.add(x).to_a
    assert_equal [2, 2], Torch.add(x, x).to_a
    assert_equal [3, 3], (x + 2).to_a
    x.add!(x)
    assert_equal [2, 2], x.to_a
  end

  def test_add_alpha
    x = Torch.tensor([1, 2, 3])
    y = Torch.tensor([10, 20, 30])
    x.add!(2, y)
    assert_equal [21, 42, 63], x.to_a
  end

  def test_mul_type
    x = Torch.tensor([1, 2, 3])
    assert_equal :int64, (x * 2).dtype
  end

  # this makes sure we override Ruby clone
  def test_clone
    x = Torch.tensor([1, 2, 3])
    y = x.clone
    x.add!(1)
    assert_equal [2, 3, 4], x.to_a
    assert_equal [1, 2, 3], y.to_a
  end

  def test_topk
    x = Torch.arange(1.0, 6.0)
    values, indices = Torch.topk(x, 3)
    assert_equal [5, 4, 3], values.to_a
    assert_equal [4, 3, 2], indices.to_a
  end

  def test_add_bad
    skip
    x = Torch.tensor([1, 2])
    Torch.add(x, 1, 1, 1)
  end

  def test_assignment
    x = Torch.tensor([1, 2, 3])
    x[1] = 0
    assert_equal [1, 0, 3], x.to_a
  end

  def test_assignment_range_index
    x = Torch.tensor([1, 2, 3])
    x[0..1] = 0
    assert_equal [0, 0, 3], x.to_a
  end

  def test_assignment_tensor_index
    skip "Not supported yet"

    x = Torch.tensor([1, 2, 3])
    x[Torch.tensor([false, true, false])] = 0
    assert_equal [1, 0, 3], x.to_a
  end

  def test_cat
    x = Torch.tensor([1, 2, 3])
    assert_equal [1, 2, 3, 1, 2, 3], Torch.cat([x, x]).to_a
  end

  def test_scalar
    x = Torch.tensor([10, 20, 30])
    assert_equal [15, 25, 35], (x + 5).to_a
    assert_equal [5, 15, 25], (x - 5).to_a
    assert_equal [50, 100, 150], (x * 5).to_a
    assert_equal [2, 4, 6], (x / 5).to_a
    assert_equal [1, 2, 0], (x % 3).to_a
    assert_equal [100, 400, 900], (x ** 2).to_a
    assert_equal [-10, -20, -30], (-x).to_a
  end

  def test_sum
    assert_equal 6, Torch.tensor([1, 2, 3]).sum.item
  end

  def test_dot
    assert_equal 7, Torch.dot(Torch.tensor([2, 3]), Torch.tensor([2, 1])).item
  end

  def test_reshape
    x = Torch.ones(6).reshape([2, 3])
    assert_equal [2, 3], x.shape
  end

  def test_argmax
    x = Torch.tensor([1, 3, 2])
    assert_equal 1, Torch.argmax(x).item
  end

  def test_eq
    x = Torch.tensor([[1, 2], [3, 4]])
    y = Torch.tensor([[1, 1], [4, 4]])
    assert_equal [[true, false], [false, true]], Torch.eq(x, y).to_a
    assert_equal [[1, 0], [0, 1]], Torch.eq(x, y).uint8.to_a
  end

  def test_flatten
    x = Torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert_equal [1, 2, 3, 4, 5, 6, 7, 8], Torch.flatten(x).to_a
    assert_equal [[1, 2, 3, 4], [5, 6, 7, 8]], Torch.flatten(x, start_dim: 1).to_a
  end

  def test_type
    x = Torch.ones([1, 2, 3])
    assert_equal :float64, x.type(:float64).dtype
    assert_equal :float64, x.double.dtype
    assert_equal :int32, x.int.dtype
    assert_equal :float32, x.dtype
  end

  def test_accessor
    x = Torch.tensor([[0, 1, 2], [3, 4, 5]])
    assert_equal [0, 1, 2], x[0].to_a
    assert_equal 5, x[1, 2].item
    assert_equal [0, 1], x[0, 0..1].to_a
    assert_equal [[[0, 1, 2], [3, 4, 5]]], x[true].to_a
    assert_equal [[[0, 1, 2], [3, 4, 5]]], x[nil].to_a
    # assert_equal [], x[false].to_a
    # if Gem::Version.new(RUBY_VERSION) >= Gem::Version.new("2.6.0")
    #   assert_equal [1, 2], x[0, eval("1..")].to_a
    # end
  end

  def test_accessor_tensor
    skip "Not implemented yet"

    x = Torch.tensor([1, 2, 3])
    index = Torch.tensor([false, true, false])
    assert_equal [2], x[index].to_a
  end

  def test_transpose
    x = Torch.randn(2, 3)
    Torch.transpose(x, 0, 1)
  end

  def test_length
    x = Torch.tensor([1, 2, 3, 5])
    assert_equal 4, x.length

    x = Torch.zeros(5, 2, 3)
    assert_equal 5, x.length
  end

  def test_save_tensor
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    tmpfile = Tempfile.new
    Torch.save(x, tmpfile.path)
    # assert_equal [[1, 2, 3], [4, 5, 6]], Torch.load(tmpfile.path).to_a
  end

  def test_masked_select
    skip "Cannot create bool from tensor method yet"

    Torch.masked_select(Torch.tensor(0), Torch.tensor(true))
  end

  def test_index_select
    x = Torch.index_select(Torch.tensor(5), 0, Torch.tensor([0]))
    assert_equal 0, x.dim
  end

  def test_exponential!
    error = assert_raises do
      Torch.empty(3).exponential!(-1.5)
    end
    assert_match "exponential_ expects lambda >= 0.0", error.message
  end
end
