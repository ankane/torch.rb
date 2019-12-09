require_relative "../test_helper"

class NormalizationLayersTest < Minitest::Test
  def test_batch_norm1d
    m = Torch::NN::BatchNorm1d.new(100)
    m = Torch::NN::BatchNorm1d.new(100, affine: false)
    input = Torch.randn(20, 100)
    output = m.call(input)
  end
end
