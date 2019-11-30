require_relative "../test_helper"

class SparseLayersTest < Minitest::Test
  # https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
  def test_embedding
    Torch.manual_seed(1)
    word_to_ix = {"hello" => 0, "world" => 1}
    embeds = Torch::NN::Embedding.new(2, 5)
    lookup_tensor = Torch.tensor([word_to_ix["hello"]], dtype: :long)
    hello_embed = embeds.call(lookup_tensor)
    assert_elements_in_delta [0.6614, 0.2669, 0.0617, 0.6213, -0.4519], hello_embed[0].to_a
  end
end
