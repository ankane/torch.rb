require_relative "../test_helper"

# this may not be needed
class FunctionalTest < Minitest::Test
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

  def test_mse_loss_different_shapes
    x = Torch.tensor([1, 2, 3]).float
    y = Torch.tensor([[1, 1, 1]]).float
    assert_output(nil, /incorrect results/) { Torch::NN::Functional.mse_loss(x, y) }
  end

  def test_one_hot
    x = Torch.tensor([0, 2, 1])
    assert_tensor [[1, 0, 0], [0, 0, 1], [0, 1, 0]], Torch::NN::Functional.one_hot(x), dtype: :int64
  end
end
