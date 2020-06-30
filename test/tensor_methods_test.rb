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

  def test_each
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = []
    x.each do |v|
      result << v.to_a
    end
    assert_equal [[1, 2, 3], [4, 5, 6]], result
  end

  def test_each_with_index
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = []
    indexes = []
    x.each_with_index do |v, i|
      result << v.to_a
      indexes << i
    end
    assert_equal [[1, 2, 3], [4, 5, 6]], result
    assert_equal [0, 1], indexes
  end

  def test_each_with_index_enumerable
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = []
    indexes = []
    x.each.with_index do |v, i|
      result << v.to_a
      indexes << i
    end
    assert_equal [[1, 2, 3], [4, 5, 6]], result
    assert_equal [0, 1], indexes
  end

  def test_map
    x = Torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert_equal [[1, 2, 3], [4, 5, 6]], x.map { |v| v.to_a }
  end

  def test_boolean_operators
    x = Torch.tensor([true, true, false, false])
    y = Torch.tensor([true, false, true, false])
    assert_equal [true, false, false, false], (x & y).to_a
    assert_equal [true, true, true, false], (x | y).to_a
    assert_equal [false, true, true, false], (x ^ y).to_a
  end
end
