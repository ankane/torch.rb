require_relative "test_helper"

class TensorMethodsTest < Minitest::Test
  # TODO test raises same error as item if many elements
  def test_to_i
    x = Torch.tensor([1.5])
    assert_equal 1, x.to_i
  end

  # TODO test raises same error as item if many elements
  def test_to_f
    x = Torch.tensor([1.5])
    assert_equal 1.5, x.to_f
  end
end
