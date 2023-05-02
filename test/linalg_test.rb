require_relative "test_helper"

class LinalgTest < Minitest::Test
  def test_norm
    a = Torch.tensor([1.0, 2, 3])
    assert_in_delta 3.7417, Torch::Linalg.norm(a).item
  end

  def test_vector_norm
    a = Torch.arange(9, dtype: :float) - 4
    b = a.reshape([3, 3])
    assert_in_delta 5.4345, Torch::Linalg.vector_norm(a, ord: 3.5).item
    assert_in_delta 5.4345, Torch::Linalg.vector_norm(b, ord: 3.5).item
  end

  def test_matrix_norm
    _a = Torch.arange(9, dtype: :float).reshape(3, 3)
    # TOOD fix
    # assert_in_delta 14.2829, Torch::Linalg.matrix_norm(a).item
    # assert_in_delta 9, Torch::Linalg.matrix_norm(a, ord: -1).item
  end

  def test_det
    a = Torch.arange(9, dtype: :float).reshape(3, 3)
    assert_in_delta 0, Torch::Linalg.det(a).item
  end
end
