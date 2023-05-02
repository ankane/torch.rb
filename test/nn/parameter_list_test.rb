require_relative "../test_helper"

class ParameterListTest < Minitest::Test
  def test_works
    param = Torch::NN::Parameter.new(Torch.randn(10, 10))
    list = Torch::NN::ParameterList.new([param])
    assert_equal 1, list.size
    assert_equal param, list[0]
    assert_kind_of Torch::NN::ParameterList, list[0..-1]
    assert_equal ["Torch::NN::Parameter"], list.map { |v| v.class.name }
  end
end
