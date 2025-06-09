require_relative "test_helper"

class OperationsTest < Minitest::Test
  def test_add
    x = Torch.ones(2)
    assert_tensor [2, 2], (x + x)
    assert_tensor [2, 2], x.add(x)
    assert_tensor [2, 2], Torch.add(x, x)
    assert_tensor [3, 3], (x + 2)
    x.add!(x)
    assert_tensor [2, 2], x
  end

  def test_add_alpha
    x = Torch.tensor([1, 2, 3])
    y = Torch.tensor([10, 20, 30])
    x.add!(y, alpha: 2)
    assert_tensor [21, 42, 63], x
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
    assert_tensor [2, 3, 4], x
    assert_tensor [1, 2, 3], y
  end

  def test_topk
    x = Torch.arange(1.0, 6.0)
    values, indices = Torch.topk(x, 3)
    assert_tensor [5, 4, 3], values
    assert_tensor [4, 3, 2], indices
  end

  def test_add_too_many_arguments
    x = Torch.tensor([1, 2])
    error = assert_raises(ArgumentError) do
      Torch.add(x, 1, 1, 1)
    end
    assert_equal "add() takes 2 positional arguments but 4 were given", error.message
  end

  def test_add_wrong_arguments
    x = Torch.tensor([1, 2])
    y = Torch.tensor([3, 4])
    error = assert_raises(ArgumentError) do
      x.add!(1, y)
    end
    assert_equal "No matching signatures", error.message
  end

  def test_cat
    x = Torch.tensor([1, 2, 3])
    assert_tensor [1, 2, 3, 1, 2, 3], Torch.cat([x, x])
  end

  def test_scalar
    x = Torch.tensor([10, 20, 30])
    assert_tensor [15, 25, 35], (x + 5)
    assert_tensor [15, 25, 35], (5 + x)
    assert_tensor [5, 15, 25], (x - 5)
    assert_tensor [-5, -15, -25], (5 - x)
    assert_tensor [50, 100, 150], (x * 5)
    assert_tensor [50, 100, 150], (5 * x)
    assert_tensor [2, 4, 6], (x / 5)
    assert_tensor [6, 3, 2], (60 / x)
    assert_tensor [1, 2, 0], (x % 3)
    assert_tensor [5, 5, 25], (25 % x)
    assert_tensor [100, 400, 900], (x ** 2)
    assert_tensor [-10, -20, -30], (-x)
  end

  def test_scalar_dtype
    x = Torch.tensor([10, 20, 30], dtype: :int16)
    assert_equal :int16, (x + 5).dtype
    assert_equal :float32, (x + 5.0).dtype
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
    assert_tensor [[true, false], [false, true]], Torch.eq(x, y)
    assert_tensor [[1, 0], [0, 1]], Torch.eq(x, y).uint8
  end

  def test_flatten
    x = Torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert_tensor [1, 2, 3, 4, 5, 6, 7, 8], Torch.flatten(x)
    assert_tensor [[1, 2, 3, 4], [5, 6, 7, 8]], Torch.flatten(x, start_dim: 1)
  end

  def test_setter_tensor_float
    x = Torch.tensor([1.0, 2, 3])
    index = Torch.tensor([false, true, false])
    x[index] = 9
    assert_tensor [1, 9, 3], x
  end

  def test_clamp
    x = Torch.tensor([1, 2, 3, 4, 5])
    x.clamp!(2, 4)
    assert_tensor [2, 2, 3, 4, 4], x
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

  def test_masked_select
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
    assert_match "exponential_ expects lambda > 0.0", error.message
  end

  def test_random!
    x = Torch.empty(10)
    assert x.random!.to_a.all? { |v| v >= 0 }
    assert x.random!(10).to_a.all? { |v| v >= 0 && v < 10 }
    assert x.random!(10, 20).to_a.all? { |v| v >= 10 && v < 20 }
  end

  def test_einsum
    x = Torch.randn(5)
    y = Torch.randn(4)
    # TODO don't require array
    z = Torch.einsum("i,j->ij", [x, y])
    assert_equal [5, 4], z.shape
  end

  def test_select
    x = Torch.arange(0, 100).view([10, 10])
    assert_tensor [3, 13, 23, 33, 43, 53, 63, 73, 83, 93], x.select(1, 3)
  end

  def test_narrow
    x = Torch.arange(0, 100).view([10, 10])
    expected = [[3, 4], [13, 14], [23, 24], [33, 34], [43, 44], [53, 54], [63, 64], [73, 74], [83, 84], [93, 94]]
    assert_tensor expected, x.narrow(1, 3, 2)
  end

  def test_hann_window
    assert_tensor [1], Torch.hann_window(1), dtype: :float32
    assert_tensor [0, 1], Torch.hann_window(2)
  end

  def test_fill_diagonal
    x = Torch.ones(2, 2)
    assert_tensor [[3, 1], [1, 3]], x.fill_diagonal!(3)
    assert_tensor [[3, 1], [1, 3]], x
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
    assert_tensor [[1], [2], [3]], x.permute([1, 0])
    assert_tensor [[1], [2], [3]], x.permute(1, 0)
  end

  def test_permute_bad
    error = assert_raises(ArgumentError) do
      Torch.tensor([[1, 2, 3]]).permute(1, 0.0)
    end
    # PyTorch returns pos 2, but pos 1 might be more intuitive
    assert_equal "permute(): argument 'dims' must be array of ints, but found element of type Float at pos 2", error.message
  end

  def test_new_full
    x = Torch.tensor([[1, 2, 3]])
    assert_tensor [[0, 0], [0, 0]], x.new_full([2, 2], 0)
  end

  def test_divide_rounding_mode_floor
    a = Torch.tensor([-1.0, 0.0, 1.0])
    b = Torch.tensor([0.0])
    c = Torch.divide(a, b, rounding_mode: "floor").to_a
    assert_equal(-Float::INFINITY, c[0])
    assert c[1].nan?
    assert_equal Float::INFINITY, c[2]
  end

  def test_divide_rounding_mode_nil
    a = Torch.full([2], 4.2)
    b = Torch.full([2], 2)
    assert_tensor [2.1, 2.1], Torch.divide(a, b, rounding_mode: nil)
  end

  def test_floor_divide
    a = Torch.tensor([4.0, -3.0])
    b = Torch.tensor([2.0, 2.0])
    assert_tensor [2.0, -2.0], Torch.floor_divide(a, b)
    assert_tensor [2.0, -1.0], Torch.div(a, b, rounding_mode: "trunc")
  end

  # TODO raise
  # RuntimeError: value cannot be converted to type int8 without overflow
  def test_tensor_overflow
    assert_tensor [-24], Torch.tensor(1000, dtype: :int8)
  end

  def test_left_shift
    assert_tensor [128], (Torch.tensor([64]) << 1)
  end

  def test_right_shift
    assert_tensor [32], (Torch.tensor([64]) >> 1)
  end

  def test_all
    assert_equal 1, Torch.all(Torch.tensor(42, dtype: :uint8), dim: 0).item
  end

  def test_conj
    x = Torch.tensor([1 + 2i])
    y = x.conj
    y.add!(2)
    # matches behavior of PyTorch
    # incorrect value in 1.10.0 release notes
    assert_tensor [3 + 2i], x
  end

  def test_conj_neg
    x = Torch.tensor([1 + 2i])
    y = x.conj
    z = y.imag
    assert z.neg?
    z.add!(2)
    assert_tensor [1 - 0i], x
  end

  def test_clamp_dtype
    a = Torch.tensor([1.0, 2.0, 3.0, 4.0], dtype: :float32)
    b = Torch.tensor([2.0, 2.0, 2.0, 2.0], dtype: :float64)
    c = Torch.tensor([3.0, 3.0, 3.0, 3.0], dtype: :float64)
    assert_equal :float64, Torch.clamp(a, b, c).dtype
  end

  def test_complex_dtype
    a = Torch.randn([2, 2], dtype: :float)
    b = Torch.tensor(1, dtype: :cdouble)

    # TODO fix
    skip
    assert_equal :complex64, (a + b).dtype
  end
end
