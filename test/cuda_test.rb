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
      assert_includes "PyTorch is not linked with support for cuda devices", error.message
    end
  end

  def test_random_seed
    if Torch::CUDA.available?
      Torch::CUDA.manual_seed_all 42

      comparables = Torch::CUDA.device_count.times.map do |i|
        x, y = 2.times.map { Torch.rand(100, device: "cuda:#{i}").to_a }
        assert x != y
        [x, y]
      end

      Torch::CUDA.manual_seed_all 42
      Torch::CUDA.device_count.times.map do |i|
        x, y = 2.times.map { Torch.rand(100, device: "cuda:#{i}").to_a }
        assert x != y
        assert_equal x, comparables[i].first
        assert_equal y, comparables[i].last
      end
    else
      error = assert_raises do
        Torch.rand 1, device: 'cuda:0'
      end

      assert_match "Could not run 'aten::empty.memory_format' with arguments from the 'CUDA' backend", error.message
    end
  end

  def test_set_device
    if Torch::CUDA.available? && Torch::CUDA.device_count.positive?
      assert_nil Torch::CUDA.set_device(0)
      error = assert_raises(ArgumentError) do
        Torch::CUDA.set_device(Torch::CUDA.device_count + 1)
      end
      assert_includes error.message, "Invalid device_id"
    else
      error = assert_raises(RuntimeError) do
        Torch::CUDA.set_device(0)
      end
      assert_includes error.message, "requires CUDA support"
    end
  end
end
