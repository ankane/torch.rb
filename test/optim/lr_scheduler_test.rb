require_relative "../test_helper"

class LRSchedulerTest < Minitest::Test
  def test_lambda_lr
    model = Net.new
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

  def test_step_lr
    model = Net.new
    optimizer = Torch::Optim::SGD.new(model.parameters, lr: 0.05)
    scheduler = Torch::Optim::LRScheduler::StepLR.new(optimizer, step_size: 1, gamma: 0.1)
    lrs = []
    3.times do
      scheduler.step
      lrs << optimizer.param_groups[0][:lr]
    end
    assert_elements_in_delta [0.005, 0.0005, 0.00005], lrs
  end
end
