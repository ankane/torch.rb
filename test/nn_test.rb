require_relative "test_helper"

class NNTest < Minitest::Test
  # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
  def test_tutorial
    net = Net.new
    assert net.inspect

    params = net.parameters
    assert_equal 10, params.size
    assert_equal [6, 1, 3, 3], params[0].size

    input = Torch.randn(1, 1, 32, 32)
    out = net.call(input)
    assert_equal [1, 10], out.size

    net.zero_grad
    out.backward(Torch.randn(1, 10))

    output = net.call(input)
    target = Torch.randn(10)
    target = target.view(1, -1)
    criterion = Torch::NN::MSELoss.new
    loss = criterion.call(output, target)

    net.zero_grad
    net.conv1.bias.grad
    loss.backward
    net.conv1.bias.grad

    learning_rate = 0.01
    net.parameters.each do |f|
      f.data.sub!(f.grad.data * learning_rate)
    end

    optimizer = Torch::Optim::SGD.new(net.parameters, lr: 0.01)
    optimizer.zero_grad
    output = net.call(input)
    loss = criterion.call(output, target)
    loss.backward
    optimizer.step
  end

  def test_to
    net = Net.new
    device = Torch::CUDA.available? ? "cuda" : "cpu"
    net.to(device)
    net.cpu
  end

  def test_dropout2d
    skip "Rand consistent with Python, dropout2d not"

    Torch.manual_seed(1)
    x = Torch.rand(2, 2)
    y = Torch::NN::Functional.dropout2d(x)
    assert_elements_in_delta [1.5153, 0.0000, 0.0000, 0.0000], y.to_a.flatten
  end

  def test_mse_loss
    x = Torch.tensor([1, 2, 3]).float
    y = Torch.tensor([1, 1, 1]).float
    assert_in_delta 5 / 3.0, Torch::NN::Functional.mse_loss(x, y).item
  end

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
