require_relative "../test_helper"

class LRSchedulerTest < Minitest::Test
  def test_lambda_lr
    model = TestNet.new
    optimizer = Torch::Optim::SGD.new(model.parameters, lr: 0.05)
    lambda1 = ->(epoch) { 0.95 ** epoch }
    scheduler = Torch::Optim::LRScheduler::LambdaLR.new(optimizer, [lambda1])
    lrs = []
    3.times do
      scheduler.step
      lrs << optimizer.param_groups[0][:lr]
    end
    assert_elements_in_delta [0.0475, 0.045125, 0.04286875], lrs
  end

  def test_multiplicative_lr
    model = TestNet.new
    optimizer = Torch::Optim::SGD.new(model.parameters, lr: 0.05)
    lambda1 = ->(epoch) { 0.95 }
    scheduler = Torch::Optim::LRScheduler::MultiplicativeLR.new(optimizer, [lambda1])
    lrs = []
    3.times do
      scheduler.step
      lrs << optimizer.param_groups[0][:lr]
    end
    assert_elements_in_delta [0.0475, 0.045125, 0.04286875], lrs
  end

  def test_step_lr
    model = TestNet.new
    optimizer = Torch::Optim::SGD.new(model.parameters, lr: 0.05)
    scheduler = Torch::Optim::LRScheduler::StepLR.new(optimizer, step_size: 1, gamma: 0.1)
    lrs = []
    3.times do
      scheduler.step
      lrs << optimizer.param_groups[0][:lr]
    end
    assert_elements_in_delta [0.005, 0.0005, 0.00005], lrs
  end

  def test_multi_step_lr
    model = TestNet.new
    optimizer = Torch::Optim::SGD.new(model.parameters, lr: 0.05)
    scheduler = Torch::Optim::LRScheduler::MultiStepLR.new(optimizer, [1, 2], gamma: 0.1)
    lrs = []
    3.times do
      scheduler.step
      lrs << optimizer.param_groups[0][:lr]
    end
    assert_elements_in_delta [0.005, 0.0005, 0.0005], lrs
  end

  def test_exponential_lr
    model = TestNet.new
    optimizer = Torch::Optim::SGD.new(model.parameters, lr: 0.05)
    scheduler = Torch::Optim::LRScheduler::ExponentialLR.new(optimizer, 0.2)
    lrs = []
    3.times do
      scheduler.step
      lrs << optimizer.param_groups[0][:lr]
    end
    assert_elements_in_delta [0.01, 0.002, 0.0004], lrs
  end

  def test_cosine_annealing_lr
    model = TestNet.new
    optimizer = Torch::Optim::SGD.new(model.parameters, lr: 0.05)
    scheduler = Torch::Optim::LRScheduler::CosineAnnealingLR.new(optimizer, 0.4)
    lrs = []
    3.times do
      scheduler.step
      lrs << optimizer.param_groups[0][:lr]
    end
    assert_elements_in_delta [0.025, 0, 0.025], lrs
  end
end
