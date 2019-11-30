require_relative "test_helper"

class LossTest < Minitest::Test
  def test_bce_loss
    skip
    assert_works Torch::NN::BCELoss, :float
  end

  def test_bce_with_logits_loss
    skip
    assert_works Torch::NN::BCEWithLogitsLoss, :float
  end

  def test_cosine_embedding_loss
    skip
    assert_works Torch::NN::CosineEmbeddingLoss, :float
  end

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

  def test_hinge_embedding_loss
    skip
    assert_works Torch::NN::HingeEmbeddingLoss, :float
  end

  def test_kl_div_loss
    skip
    assert_works Torch::NN::KLDivLoss, :float
  end

  def test_l1_loss
    assert_works Torch::NN::L1Loss, :float
  end

  def test_margin_ranking_loss
    skip
    assert_works Torch::NN::MarginRankingLoss, :float
  end

  def test_mse_loss
    assert_works Torch::NN::MSELoss, :float
  end

  def test_multi_label_margin_loss
    skip
    assert_works Torch::NN::MultiLabelMarginLoss, :float
  end

  def test_multi_label_soft_margin_loss
    skip
    assert_works Torch::NN::MultiLabelSoftMarginLoss, :float
  end

  def test_multi_margin_loss
    skip
    assert_works Torch::NN::MultiMarginLoss, :float
  end

  def test_nll_loss
    assert_works Torch::NN::NLLLoss, :long
  end

  def test_poisson_nll_loss
    assert_works Torch::NN::PoissonNLLLoss, :float
  end

  def test_smooth_l1_loss
    skip
    assert_works Torch::NN::SmoothL1Loss, :float
  end

  def test_soft_margin_loss
    skip
    assert_works Torch::NN::SoftMarginLoss, :float
  end

  def test_triplet_margin_loss
    skip
    assert_works Torch::NN::TripletMarginLoss, :float
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
