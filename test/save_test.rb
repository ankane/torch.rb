require_relative "test_helper"

class SaveTest < Minitest::Test
  def test_nil
    assert_save nil
  end

  def test_bool
    assert_save true
    assert_save false
  end

  def test_integer
    assert_save 123
  end

  def test_integer_out_of_range
    tmpfile = Tempfile.new
    error = assert_raises(RangeError) do
      Torch.save(2**64, tmpfile.path)
    end
    assert_match "bignum too big to convert into `long", error.message
  end

  def test_float
    assert_save 1.23
  end

  def test_string
    assert_save "hello"
  end

  def test_tensor
    assert_save Torch.tensor([[1, 2, 3], [4, 5, 6]])
  end

  def test_list_tensor
    x = Torch.tensor([1, 2, 3])
    y = Torch.tensor([4, 5, 6])
    assert_save [x, y]
  end

  def test_list_integer
    assert_save [1, 2, 3]
  end

  def test_hash
    assert_save({"hello" => 1, "world" => 2})
  end

  private

  def assert_save(obj)
    tmpfile = Tempfile.new
    Torch.save(obj, tmpfile.path)
    act = Torch.load(tmpfile.path)

    if obj.is_a?(Torch::Tensor)
      assert_equal obj.to_a, act.to_a
    elsif obj.is_a?(Array) && obj.first.is_a?(Torch::Tensor)
      assert_equal obj.map(&:to_a), act.map(&:to_a)
    elsif obj.nil?
      assert_nil act
    else
      assert_equal obj, act
    end
  end
end
