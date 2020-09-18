require_relative "../test_helper"

class VisionTest < Minitest::Test
  def test_upsample
    x = Torch.tensor([[[1.0]]])
    assert_equal [[[1, 1]]], Torch::NN::Upsample.new(scale_factor: 2).call(x).to_a
  end
end
