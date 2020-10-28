require_relative "test_helper"

class AutogradTest < Minitest::Test
  def test_example
    x = Torch.ones(2, 2, requires_grad: true)
    y = x + 2
    z = y * y * 3
    out = z.mean
    out.backward
    assert_equal [[4.5, 4.5], [4.5, 4.5]], x.grad.to_a
  end

  def test_example_backward
    x = Torch.randn(3, requires_grad: true)

    y = x * 2
    while y.data.norm < 1000
      y = y * 2
    end

    v = Torch.tensor([0.1, 1.0, 0.0001], dtype: :float)
    y.backward(v)

    assert_equal [3], x.grad.size
  end

  def test_requires_grad
    a = Torch.randn(2, 2)
    a = ((a * 3) / (a - 1))
    assert !a.requires_grad
    a.requires_grad!(true)
    assert a.requires_grad
    b = (a * a).sum
  end

  def test_no_grad
    x = Torch.tensor([1.0], requires_grad: true)
    y = nil
    Torch.no_grad do
      y = x * 2
      assert !Torch.grad_enabled?
    end
    assert !y.requires_grad
    assert Torch.grad_enabled?
  end

  def test_enable_grad
    assert Torch.grad_enabled?
    Torch.no_grad do
      assert !Torch.grad_enabled?
      Torch.enable_grad do
        assert Torch.grad_enabled?
      end
    end
  end

  def test_grad
    x = Torch.tensor([1, 2, 3])
    assert_nil x.grad
  end

  def test_set_grad
    x = Torch.tensor([1, 2, 3])
    x.grad = Torch.tensor([1, 1, 1])
    assert_equal [1, 1, 1], x.grad.to_a
  end

  def test_variable
    x = Torch.tensor([1, 2, 3])
    v = nil
    assert_output(nil, /deprecated/) do
      v = Torch::Autograd::Variable.new(x)
    end
    assert x.eql?(v)
  end

  def test_variable_invalid
    error = assert_raises(ArgumentError) do
      Torch::Autograd::Variable.new(Object.new)
    end
    assert_equal "Variable data has to be a tensor, but got Object", error.message
  end

  # 1.7.0 behavior
  def test_max
    a = Torch.tensor([3.0, 2, 3], requires_grad: true)
    a.max.backward
    # TODO debug
    # assert_equal [0.5, 0, 0.5], a.grad.to_a
  end
end
