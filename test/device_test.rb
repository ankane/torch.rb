require_relative "test_helper"

class DeviceTest < Minitest::Test
  def test_equal
    a = Torch.device("cpu")
    b = Torch.device("cpu")
    c = Torch.device("cpu:1")
    assert_equal a, b
    refute_equal a, c
  end
end
