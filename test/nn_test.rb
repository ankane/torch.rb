require_relative "test_helper"

class NNTest < Minitest::Test
  def test_works
    net = Net.new

    params = net.parameters
    assert_equal 10, params.size
    assert_equal [6, 1, 3, 3], params[0].size

    input = Torch.randn(1, 1, 32, 32)
    out = net.call(input)
    assert_equal [1, 10], out.size
  end
end
