require_relative "test_helper"

class LossTest < Minitest::Test
  def test_cross_entropy_loss
    assert_works Torch::NN::CrossEntropyLoss, :long
  end

  def test_l1_loss
    assert_works Torch::NN::L1Loss, :float
  end

  def test_mse_loss
    assert_works Torch::NN::MSELoss, :float
  end

  def test_nll_loss
    assert_works Torch::NN::NLLLoss, :long
  end

  private

  def assert_works(cls, dtype)
    loss = cls.new
    input = Torch.randn(3, 5, requires_grad: true)
    target = dtype == :float ? Torch.randn(3, 5) : Torch.tensor([1, 0, 4])
    output = loss.call(input, target)
    output.backward
  end
end
