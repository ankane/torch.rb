require_relative "../test_helper"

class ModuleListTest < Minitest::Test
  def test_works
    mod = Torch::NN::Linear.new(2, 2)
    list = Torch::NN::ModuleList.new([mod])
    assert_equal 1, list.size
    assert_equal mod, list[0]
    assert_kind_of Torch::NN::ModuleList, list[0..-1]
    assert_equal ["Torch::NN::Linear"], list.map { |v| v.class.name }
  end
end
