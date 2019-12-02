require_relative "../test_helper"

class ModuleTest < Minitest::Test
  def test_modules
    assert_equal 6, net.modules.size
    assert_equal 6, net.named_modules.size
    assert_equal ["", "conv1", "conv2", "fc1", "fc2", "fc3"], net.named_modules.keys
  end

  def test_parameters
    assert_equal 10, net.parameters.size
    assert_equal 10, net.named_parameters.size
    expected = %w(
      conv1.weight conv1.bias conv2.weight conv2.bias fc1.weight fc1.bias
      fc2.weight fc2.bias fc3.weight fc3.bias
    )
    assert_equal expected, net.named_parameters.keys
  end

  def test_accessors
    assert net.conv1.weight
    assert net.conv1.bias
    assert net.fc1.weight
    assert net.fc1.bias
  end

  private

  def net
    @net ||= Net.new
  end
end
