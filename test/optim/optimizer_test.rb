require_relative "../test_helper"

class OptimizerTest < Minitest::Test
  def test_adadelta
    assert_works Torch::Optim::Adadelta, weight_decay: 0.1
  end

  def test_adagrad
    assert_works Torch::Optim::Adagrad, weight_decay: 0.1
  end

  def test_adam
    assert_works Torch::Optim::Adam, weight_decay: 0.1
    assert_works Torch::Optim::Adam, amsgrad: true
  end

  def test_adamax
    assert_works Torch::Optim::Adamax, weight_decay: 0.1
  end

  def test_adamw
    assert_works Torch::Optim::AdamW
    assert_works Torch::Optim::AdamW, amsgrad: true
  end

  def test_asgd
    assert_works Torch::Optim::ASGD, weight_decay: 0.1
  end

  def test_rmsprop
    assert_works Torch::Optim::RMSprop
    assert_works Torch::Optim::RMSprop, weight_decay: 0.1, centered: true, momentum: 0.1
  end

  def test_rprop
    assert_works Torch::Optim::Rprop
  end

  def test_sgd
    assert_works Torch::Optim::SGD, momentum: 0.8, nesterov: true
  end

  private

  def assert_works(cls, **options)
    net = TestNet.new
    optimizer = cls.new(net.parameters, lr: 0.01, **options)
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

    # tmpfile = Tempfile.new
    # Torch.save(optimizer.state_dict, tmpfile.path)
    # state_dict = Torch.load(tmpfile.path)
    # assert_equal optimizer.state_dict, state_dict

    # optimizer.load_state_dict(state_dict)
  end
end
