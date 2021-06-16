require_relative "test_helper"

class FFTTest < Minitest::Test
  def test_fft
    t = Torch.arange(4)
    assert_equal [6, -2+2i, -2, -2-2i], Torch::FFT.fft(t).to_a
  end

  def test_ifft
    t = Torch.tensor([6, -2+2i, -2, -2-2i])
    assert_equal [0, 1, 2, 3], Torch::FFT.ifft(t).to_a
  end
end
