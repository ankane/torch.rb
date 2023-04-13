require_relative "test_helper"

class BackendsTest < Minitest::Test
  def test_openmp
    Torch::Backends::OpenMP.available?
  end

  def test_mkl
    Torch::Backends::MKL.available?
  end

  def test_mps
    Torch::Backends::MPS.available?
  end
end
