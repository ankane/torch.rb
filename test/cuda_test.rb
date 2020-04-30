require_relative "test_helper"

class CUDATest < Minitest::Test
  def test_available
    assert !Torch::CUDA.available?.nil?
    assert Torch::CUDA.device_count
  end

  def test_device
    device = Torch.device("cpu")
    assert_equal "cpu", device.type
    assert !device.index?
  end

  def test_tensor
    x = Torch.tensor([1, 2, 3])

    if Torch::CUDA.available?
      assert_equal [1, 2, 3], x.cuda.to_a
    else
      error = assert_raises do
        x.cuda
      end
      # TODO see why this differs from PyTorch
      # Torch not compiled with CUDA enabled
      assert_equal "PyTorch is not linked with support for cuda devices", error.message
    end
  end
end
