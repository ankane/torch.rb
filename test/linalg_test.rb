require_relative "test_helper"

class LinalgTest < Minitest::Test
  def test_norm
    x = Torch.tensor([1.0, 2, 3])
    assert_in_delta 3.7417, Torch::Linalg.norm(x).item
  end
end
