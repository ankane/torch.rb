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
end
