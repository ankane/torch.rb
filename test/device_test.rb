require_relative "test_helper"

class DeviceTest < Minitest::Test
  def test_type
    assert_equal Torch.device("cpu:0").type, "cpu"
  end

  def test_index
    assert_nil Torch.device("cpu").index
    assert_equal 0, Torch.device("cpu:0").index
  end

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

  def test_to_s
    assert_equal "cpu", Torch.device("cpu").to_s
  end
end
