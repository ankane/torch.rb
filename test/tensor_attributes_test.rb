require_relative "test_helper"

class TensorAttributesTest < Minitest::Test
  def test_dim
    x = Torch.empty(3, 4, 5)
    assert_equal 3, x.dim
    assert_equal 3, x.ndim
    assert_equal 3, x.ndimension
  end

  def test_dtype
    %i(uint8 int8 int16 int32 int64 float32 float64).each do |dtype|
      x = Torch.tensor([1, 2, 3], dtype: dtype)
      assert_tensor [1, 2, 3], x, dtype: dtype
    end
  end

  def test_dtype_complex
    # TODO support complex32
    %i(complex64 complex128).each do |dtype|
      x = Torch.tensor([1i, 2+3i], dtype: dtype)
      assert_tensor [1i, 2+3i], x, dtype: dtype
    end
  end

  def test_dtype_bool
    x = Torch.tensor([false, true, false])
    assert_tensor [false, true, false], x, dtype: :bool
  end

  def test_dtype_default
    assert_equal :int64, Torch.tensor([1]).dtype
    assert_equal :float32, Torch.tensor([1.0]).dtype
    assert_equal :complex64, Torch.tensor([1i]).dtype
  end

  def test_dtype_numo
    x = Torch.tensor(Numo::DFloat.asarray([1, 2, 3]))
    assert_equal :float64, x.dtype

    x = Torch.tensor(Numo::SFloat.asarray([1, 2, 3]))
    assert_equal :float32, x.dtype

    x = Torch.tensor(Numo::Int64.asarray([1, 2, 3]))
    assert_equal :int64, x.dtype

    x = Torch.tensor(Numo::Int32.asarray([1, 2, 3]))
    assert_equal :int32, x.dtype
  end

  # different message locally and on CI
  def test_dtype_bad
    assert_raises(TypeError) do
      Torch.tensor([true], dtype: :int64)
    end
  end

  # TODO improve error type and message
  def test_dtype_bad_complex
    error = assert_raises(NoMethodError) do
      Torch.tensor([true], dtype: :complex64)
    end
    assert_match(/undefined method [`']real' for true/, error.message)
  end

  def test_layout
    # TODO support sparse
    %i(strided).each do |layout|
      x = Torch.tensor([1, 2, 3], layout: layout)
      assert_equal layout, x.layout
      assert_tensor [1, 2, 3], x
    end
  end

  def test_layout_bad
    error = assert_raises do
      Torch.tensor([1, 2, 3], layout: "bad")
    end
    assert_equal "Unsupported layout: bad", error.message
  end

  def test_device
    x = Torch.tensor([1, 2, 3], device: "cpu")
    assert_equal "cpu", x.device
  end

  def test_device_bad
    error = assert_raises do
      Torch.tensor([1, 2, 3], device: "bad")
    end
    assert_match(/Expected one of .+ device type at start of device string: bad/, error.message)
  end

  def test_requires_grad
    x = Torch.ones(2, 3)
    assert !x.requires_grad?
    x.requires_grad!
    assert x.requires_grad?
  end

  def test_accessor_methods
    x = Torch.ones(2, 3)
    assert_equal :float32, x.dtype
    assert_equal :strided, x.layout
    assert_equal "cpu", x.device
    assert_equal 6, x.numel
    assert_equal 4, x.element_size
    assert !x.cuda?
    assert !x.distributed?
    assert !x.complex?
    assert x.floating_point?
    assert x.signed?
    assert !x.sparse?
    assert !x.quantized?
  end

  def test_shape
    x = Torch.ones(2, 3)
    assert_equal [2, 3], x.shape
  end

  def test_size
    x = Torch.ones(2, 3)
    assert_equal [2, 3], x.shape
    assert_equal [2, 3], x.size
    assert_equal 2, x.size(0)
    assert_equal 3, x.size(1)
  end

  def test_stride
    x = Torch.ones(2, 3)
    assert_equal [3, 1], x.stride
    assert_equal 3, x.stride(0)
    assert_equal 1, x.stride(1)
  end

  def test_inspect
    assert_equal "tensor(5.)", Torch.tensor(5.0).inspect
    assert_equal "tensor([])", Torch.tensor([]).inspect
    assert_equal "tensor([1., 1., 1., 1., 1.])", Torch.ones(5).inspect
    assert_equal "tensor([ 1.0000, -1.2500], requires_grad: true)", Torch.tensor([1, -1.25], requires_grad: true).inspect
    assert_equal "tensor([Inf])", Torch.tensor([100.0]).exp.inspect
  end

  def test_tensor
    assert Torch.tensor?(Torch.empty(1))
    assert !Torch.tensor?(1)
  end

  def test_floating_point
    assert Torch.floating_point?(Torch.empty(1))
    assert !Torch.floating_point?(Torch.empty(1, dtype: Torch.long))
  end

  def test_inconsistent_dimensions
    error = assert_raises(Torch::Error) do
      Torch.tensor([[1, 2], [3]])
    end
    assert_equal "Inconsistent dimensions", error.message
  end

  # TODO raise error
  def test_inconsistent_dimensions_correct_size
    x = Torch.tensor([[1, 2], [3], [4, 5, 6]])
    assert_tensor [[1, 2], [3, 4], [5, 6]], x
  end
end
