require_relative "test_helper"

class RandomTest < Minitest::Test
  def test_initial_seed
    assert Torch::Random.respond_to?(:initial_seed)
  end
end
