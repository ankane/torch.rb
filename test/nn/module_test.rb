require_relative "../test_helper"

class ModuleTest < Minitest::Test
  def test_parameters
    assert_equal 10, net.parameters.size
    assert_equal 10, net.named_parameters.size
    expected = %w(
      conv1.weight conv1.bias conv2.weight conv2.bias fc1.weight
      fc1.bias fc2.weight fc2.bias fc3.weight fc3.bias
    )
    assert_equal expected, net.named_parameters.keys
    assert_includes net.parameters[0].inspect, "Parameter containing"
  end

  def test_buffers
    assert_equal 0, net.buffers.size
    assert_equal 0, net.named_buffers.size
  end

  def test_children
    assert_equal 5, net.children.size
    assert_equal 5, net.named_children.size
    assert_equal %w(conv1 conv2 fc1 fc2 fc3), net.named_children.keys
  end

  def test_modules
    assert_equal 6, net.modules.size
    assert_equal 6, net.named_modules.size
    assert_equal [""] + %w(conv1 conv2 fc1 fc2 fc3), net.named_modules.keys
  end

  def test_accessors
    assert net.conv1.weight
    assert net.conv1.bias
    assert net.fc1.weight
    assert net.fc1.bias
  end

  def test_to
    net = TestNet.new
    device = Torch::CUDA.available? ? "cuda" : "cpu"
    net.to(device)
    net.cpu
  end

  def test_state_dict
    net = TestNet.new
    assert_equal 10, net.state_dict.size

    tmpfile = Tempfile.new
    Torch.save(net.state_dict, tmpfile.path)

    net = TestNet.new
    net.load_state_dict(Torch.load(tmpfile.path))
    net.eval
  end

  def test_inspect
    assert_match "(conv1): Conv2d(1, 6, kernel_size: [3, 3], stride: [1, 1])", net.inspect
  end

  def test_apply_buffers
    mod = Torch::NN::BatchNorm.new(1)
    assert_equal :float32, mod.running_mean.dtype
    assert_equal :float32, mod.named_buffers["running_mean"].dtype
    assert_equal :float32, mod.instance_variable_get(:@running_mean).dtype
    mod.half
    assert_equal :float16, mod.running_mean.dtype
    assert_equal :float16, mod.named_buffers["running_mean"].dtype
    assert_equal :float16, mod.instance_variable_get(:@running_mean).dtype
  end

  def test_load_state_dict
    skip "Not working yet"

    net = Torch::NN::Linear.new(10, 2)
    net.load_state_dict(net.state_dict)
  end

  def test_load_state_dict_unknown_parameter
    net = TestNet.new
    error = assert_raises(Torch::Error) do
      net.load_state_dict({"conv1.bad_parameter" => 1})
    end
    assert_equal "Unknown parameter `bad_parameter` in module `conv1`", error.message
  end

  def test_load_state_dict_unknown_module
    net = TestNet.new
    error = assert_raises(Torch::Error) do
      net.load_state_dict({"bad_module.bad_parameter" => 1})
    end
    assert_equal "Unknown module: bad_module", error.message
  end

  def test_load_state_dict_missing_keys
    skip "Not working yet"

    net = TestNet.new
    error = assert_raises(Torch::Error) do
      net.load_state_dict({})
    end
    assert_equal "Missing key(s) in state_dict: weight, bias", error.message
  end

  def test_load_state_dict_unexpected_keys
    skip "Not working yet"

    net = TestNet.new
    state_dict = net.state_dict
    state_dict["bad_key"] = 1
    error = assert_raises(Torch::Error) do
      net.load_state_dict(state_dict)
    end
    assert_equal "Unexpected key(s) in state_dict: bad_key", error.message
  end

  private

  def net
    @net ||= TestNet.new
  end
end
