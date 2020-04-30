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
    net = Net.new
    device = Torch::CUDA.available? ? "cuda" : "cpu"
    net.to(device)
    net.cpu
  end

  def test_state_dict
    net = Net.new
    assert_equal 10, net.state_dict.size

    # puts "Model's state_dict:"
    # net.state_dict.each do |k, v|
    #   puts "#{k}\t#{v.size}"
    # end

    tmpfile = Tempfile.new
    Torch.save(net.state_dict, tmpfile.path)

    net = Net.new
    net.load_state_dict(Torch.load(tmpfile.path))
    net.eval

    # optimizer = Torch::Optim::SGD.new(net.parameters, lr: 0.001, momentum: 0.9)

    # puts "Optimizer's state_dict:"
    # optimizer.state_dict.each do |k, v|
    #   puts "#{k}\t#{v}"
    # end

    # tmpfile2 = Tempfile.new
    # Torch.save(optimizer.state_dict, tmpfile2.path)
  end

  def test_inspect
    assert_match "(conv1): Conv2d(1, 6, kernel_size: [3, 3], stride: [1, 1])", net.inspect
  end

  private

  def net
    @net ||= Net.new
  end
end
