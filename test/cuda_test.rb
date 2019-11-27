require_relative "test_helper"

class CudaTest < Minitest::Test
  def test_works
    Torch::CUDA.available?
  end
end
