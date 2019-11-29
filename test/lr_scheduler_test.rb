require_relative "test_helper"

class LRSchedulerTest < Minitest::Test
  def test_step_lr
    skip "Step not implemented yet"

    model = Net.new
    optimizer = Torch::Optim::SGD.new(model.parameters, lr: 0.05)
    scheduler = Torch::Optim::LRScheduler::StepLR.new(optimizer, step_size: 1, gamma: 0.1)
    lrs = []
    3.times do |epoch|
      scheduler.step
      lrs << optimizer.param_groups[0][:lr]
    end
    assert_elements_in_delta [0.05, 0.005, 0.0005], lrs
  end
end
