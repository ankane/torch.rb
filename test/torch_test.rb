require_relative "test_helper"

class TorchTest < Minitest::Test
  def test_show_config
    config = Torch.show_config
    assert_match "PyTorch built with:", config
    # pre-built Mac library has
    # OpenMP disabled in include/ATen/Config.h
    # starting with LibTorch 1.13.0
    assert_match "USE_OPENMP=ON", config unless mac?
  end

  def test_parallel_info
    info = Torch.parallel_info
    assert_match "ATen/Parallel:", info
    # pre-built Mac library has
    # OpenMP disabled in include/ATen/Config.h
    # starting with LibTorch 1.13.0
    assert_match "ATen parallel backend: OpenMP", info unless mac?
  end

  def test_tutorial
    x = Torch.empty(5, 3)

    x = Torch.rand(5, 3)

    x = Torch.zeros(5, 3, dtype: :long)
    assert_equal :int64, x.dtype

    x = Torch.tensor([5.5, 3])

    x = x.new_ones(5, 3, dtype: :double)
    assert_equal :float64, x.dtype

    x = Torch.randn_like(x, dtype: :float)
    assert_equal :float32, x.dtype

    assert_equal [5, 3], x.size

    y = Torch.rand(5, 3)
    _ = x + y

    Torch.add(x, y)

    result = Torch.empty(5, 3)
    Torch.add(x, y, out: result)

    y.add!(x)

    x[0]

    x = Torch.randn(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)
    assert_equal [4, 4], x.size
    assert_equal [16], y.size
    assert_equal [2, 8], z.size

    assert_equal 16, x.numel

    x = Torch.randn(1)
    x.item

    a = Torch.ones(5)
    b = a.numo
    assert_kind_of Numo::SFloat, b
  end

  def test_friendly_error_tensor_no_cpp_trace
    error = assert_raises(Torch::Error) do
      Torch.arange(0, 100).view([10, 10]).select(2, 0)
    end
    assert_equal "Dimension out of range (expected to be in range of [-2, 1], but got 2)", error.message
  end

  def test_friendly_error_torch_no_cpp_trace
    error = assert_raises(Torch::Error) do
      Torch.select(Torch.arange(0, 100).view([10, 10]), 2, 0)
    end
    assert_equal "Dimension out of range (expected to be in range of [-2, 1], but got 2)", error.message
  end

  def test_byte_storage
    x = stress_gc do
      s = Torch::ByteStorage.from_buffer("\x01\x02\x03")
      Torch::ByteTensor.new(s)
    end
    assert_equal [1, 2, 3], x.to_a
  end
end
