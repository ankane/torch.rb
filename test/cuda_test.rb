require_relative "test_helper"

class CUDATest < Minitest::Test
  def test_works
    Torch::CUDA.available?
  end
end
