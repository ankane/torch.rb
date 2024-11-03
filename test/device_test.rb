require_relative "test_helper"

class DeviceTest < Minitest::Test
  def test_equal
    a = Torch.device("cpu")
    b = Torch.device("cpu")
    c = Torch.device("cpu:0")
    assert_equal a, b
    refute_equal a, c
  end

  def test_inspect
    assert_equal %!device(type: "cpu")!, Torch.device("cpu").inspect
    assert_equal %!device(type: "cpu", index: 0)!, Torch.device("cpu:0").inspect
  end
end
