require_relative "test_helper"

class SaveTest < Minitest::Test
  def test_nil
    assert_save nil
  end

  def test_bool
    assert_save true
    assert_save false
  end

  # TODO test out of range
  def test_integer
    assert_save 123
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

  def test_hash
    # currently broken in LibTorch
    # assert_save({"hello" => 1, "world" => 2})
  end

  private

  def assert_save(obj)
    tmpfile = Tempfile.new
    Torch.save(obj, tmpfile.path)
    act = Torch.load(tmpfile.path)

    if obj.is_a?(Torch::Tensor)
      assert_equal obj.to_a, act.to_a
    elsif obj.nil?
      assert_nil act
    else
      assert_equal obj, act
    end
  end
end
