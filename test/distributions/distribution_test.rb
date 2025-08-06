require_relative "../test_helper"

class DistributionTest < Minitest::Test
  def test_normal
    m = Torch::Distributions::Normal.new(Torch.tensor([0.0]), Torch.tensor([1.0]))
    assert_equal [1], m.sample.shape
    assert_equal [2, 1], m.sample(sample_shape: [2]).shape
  end
end
