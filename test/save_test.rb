require_relative "test_helper"

class SaveTest < Minitest::Test
  def test_nil
    assert_save nil
  end

  def test_bool
    assert_save true
    assert_save false
  end

  def test_integer
    assert_save 123
  end

  def test_integer_out_of_range
    tmpfile = Tempfile.new
    error = assert_raises(RangeError) do
      Torch.save(2**64, tmpfile.path)
    end
    assert_match(/bignum too big to convert into [`']long/, error.message)
  end

  def test_float
    assert_save 1.23
  end

  def test_string
    assert_save "hello"
  end

  def test_tensor
    assert_save Torch.tensor([[1, 2, 3], [4, 5, 6]])
  end

  def test_list_tensor
    x = Torch.tensor([1, 2, 3])
    y = Torch.tensor([4, 5, 6])
    assert_save [x, y]
  end

  def test_list_integer
    assert_save [1, 2, 3]
  end

  def test_hash
    assert_save({"hello" => 1, "world" => 2})
  end

  def test_load_missing
    error = assert_raises(Errno::ENOENT) do
      Torch.load("missing.bin")
    end
    assert_equal "No such file or directory @ rb_sysopen - missing.bin", error.message
  end

  def test_load_with_map_location_string
    tmpfile = Tempfile.new
    tensor = Torch.tensor([1, 2, 3])
    Torch.save(tensor, tmpfile.path)
    loaded = Torch.load(tmpfile.path, map_location: "cpu")
    assert_equal tensor.to_a, loaded.to_a
  end

  def test_load_with_map_location_callable
    tmpfile = Tempfile.new
    tensor = Torch.tensor([1, 2, 3])
    Torch.save(tensor, tmpfile.path)
    seen = []
    loaded = Torch.load(tmpfile.path, map_location: lambda { |value, loc|
      seen << loc
      value
    })
    assert_equal tensor.to_a, loaded.to_a
    assert_equal ["cpu"], seen
  end

  def test_load_with_weights_only
    tmpfile = Tempfile.new
    tensor = Torch.tensor([1, 2, 3])
    Torch.save(tensor, tmpfile.path)
    loaded = Torch.load(tmpfile.path, weights_only: true)
    assert_equal tensor.to_a, loaded.to_a
  end

  def test_load_map_location_cuda_to_cpu
    skip "Requires CUDA" unless Torch::CUDA.available?

    tmpfile = Tempfile.new
    tensor = Torch.tensor([1, 2, 3]).cuda
    Torch.save(tensor, tmpfile.path)

    loaded = Torch.load(tmpfile.path, map_location: "cpu")
    assert_equal "cpu", loaded.device.type
    assert_equal tensor.cpu.to_a, loaded.to_a
  end

  def test_load_map_location_cpu_to_cuda
    skip "Requires CUDA" unless Torch::CUDA.available?

    tmpfile = Tempfile.new
    tensor = Torch.tensor([1, 2, 3])
    Torch.save(tensor, tmpfile.path)

    device = "cuda:0"
    loaded = Torch.load(tmpfile.path, map_location: device)
    assert_equal "cuda", loaded.device.type
    assert_equal 0, loaded.device.index
    assert_equal tensor.to_a, loaded.cpu.to_a
  end

  private

  def assert_save(obj)
    tmpfile = Tempfile.new
    Torch.save(obj, tmpfile.path)
    act = Torch.load(tmpfile.path)

    if obj.is_a?(Torch::Tensor)
      assert_equal obj.to_a, act.to_a
    elsif obj.is_a?(Array) && obj.first.is_a?(Torch::Tensor)
      assert_equal obj.map(&:to_a), act.map(&:to_a)
    elsif obj.nil?
      assert_nil act
    else
      assert_equal obj, act
    end
  end
end
