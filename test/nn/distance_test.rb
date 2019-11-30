require_relative "../test_helper"

class DistanceTest < Minitest::Test
  def test_cosine_similarity
    input1 = Torch.randn(100, 128)
    input2 = Torch.randn(100, 128)
    cos = Torch::NN::CosineSimilarity.new(dim: 1, eps: 1e-6)
    output = cos.call(input1, input2)
  end

  def test_pairwise_distance
    pdist = Torch::NN::PairwiseDistance.new(p: 2)
    input1 = Torch.randn(100, 128)
    input2 = Torch.randn(100, 128)
    output = pdist.call(input1, input2)
  end
end
