require_relative "../test_helper"

class SparseLayersTest < Minitest::Test
  def test_embedding
    embedding = Torch::NN::Embedding.new(10, 3)
    input = Torch::LongTensor.new([[1, 2, 4, 5], [4, 3, 2, 9]])
    embedding.call(input)

    embedding = Torch::NN::Embedding.new(10, 3, padding_idx: 0)
    input = Torch::LongTensor.new([[0, 2, 0, 5]])
    embedding.call(input)
  end

  def test_one_hot
    labels = Torch.tensor([0, 2, 1])
    assert_tensor [[1, 0, 0], [0, 0, 1], [0, 1, 0]], Torch::NN::Functional.one_hot(labels)
    assert_tensor [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], Torch::NN::Functional.one_hot(labels, num_classes: 4)
    assert_equal :int64, Torch::NN::Functional.one_hot(labels).dtype
    assert_equal [2, 2, 3], Torch::NN::Functional.one_hot(Torch.tensor([[0, 1], [2, 0]])).shape
  end

  def test_embedding_bag
    embedding_sum = Torch::NN::EmbeddingBag.new(10, 3, mode: "sum")
    input = Torch::LongTensor.new([1, 2, 4, 5, 4, 3, 2, 9])
    offsets = Torch::LongTensor.new([0, 4])
    embedding_sum.call(input, offsets: offsets)
  end

  def test_embedding_bag_from_pretrained
    skip "Not implemented yet"

    weight = Torch::FloatTensor.new([[1, 2.3, 3], [4, 5.1, 6.3]])
    embeddingbag = Torch::NN::EmbeddingBag.from_pretrained(weight)
    input = Torch::LongTensor.new([[1, 0]])
    embeddingbag.call(input)
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
