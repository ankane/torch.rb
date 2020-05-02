require_relative "../test_helper"

class OptimizerTest < Minitest::Test
  def test_adadelta
    assert_works Torch::Optim::Adadelta
  end

  def test_adagrad
    assert_works Torch::Optim::Adagrad
  end

  def test_adam
    assert_works Torch::Optim::Adam
  end

  def test_adamax
    assert_works Torch::Optim::Adamax
  end

  def test_adamw
    assert_works Torch::Optim::AdamW
  end

  def test_asgd
    assert_works Torch::Optim::ASGD
  end

  def test_rmsprop
    assert_works Torch::Optim::RMSprop
  end

  def test_rprop
    skip "Need to implement []="
    assert_works Torch::Optim::Rprop
  end

  def test_sgd
    assert_works Torch::Optim::SGD
  end

  private

  def assert_works(cls)
    net = TestNet.new
    optimizer = cls.new(net.parameters, lr: 0.01)
    input = Torch.randn(1, 1, 32, 32)
    target = Torch.randn(10)
    target = target.view(1, -1)
    criterion = Torch::NN::MSELoss.new

    # run a few times for conditional logic
    3.times do
      optimizer.zero_grad
      output = net.call(input)
      loss = criterion.call(output, target)
      loss.backward
      optimizer.step
    end

    state_dict = optimizer.state_dict
    # optimizer.load_state_dict(state_dict)
  end
end
