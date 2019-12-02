require_relative "../test_helper"

class SparseLayersTest < Minitest::Test
  def test_embedding
    embedding = Torch::NN::Embedding.new(10, 3)
    input = Torch::LongTensor.new([[1,2,4,5],[4,3,2,9]])
    embedding.call(input)

    embedding = Torch::NN::Embedding.new(10, 3, padding_idx: 0)
    input = Torch::LongTensor.new([[0,2,0,5]])
    embedding.call(input)
  end

  # https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
  def test_embedding_tutorial
    Torch.manual_seed(1)
    word_to_ix = {"hello" => 0, "world" => 1}
    embeds = Torch::NN::Embedding.new(2, 5)
    lookup_tensor = Torch.tensor([word_to_ix["hello"]], dtype: :long)
    hello_embed = embeds.call(lookup_tensor)
    assert_elements_in_delta [0.6614, 0.2669, 0.0617, 0.6213, -0.4519], hello_embed[0].to_a
  end
end
