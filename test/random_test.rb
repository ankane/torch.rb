require_relative "test_helper"

class RandomTest < Minitest::Test
  def test_initial_seed
    Torch.manual_seed(123)
    assert_equal 123, Torch::Random.initial_seed
  end
end
