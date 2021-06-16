require_relative "test_helper"

class FFTTest < Minitest::Test
  def test_fft
    t = Torch.arange(4)
    assert_equal [6+0i, -2+2i, -2+0i, -2-2i], Torch::FFT.fft(t).to_a
  end
end
