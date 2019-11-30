require_relative "test_helper"

class LossTest < Minitest::Test
  def test_cross_entropy_loss
    assert_works Torch::NN::CrossEntropyLoss, :long
  end

  def test_ctc_loss
    t = 50
    c = 20
    n = 16
    s = 30
    s_min = 10

    input = Torch.randn(t, n, c).log_softmax(2).detach.requires_grad!
    target = Torch.randint(1, c, [n, s], dtype: :long)

    input_lengths = Torch.full([n], t, dtype: :long)
    target_lengths = Torch.randint(s_min, s, [n], dtype: :long)
    ctc_loss = Torch::NN::CTCLoss.new
    loss = ctc_loss.call(input, target, input_lengths, target_lengths)
    loss.backward
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
