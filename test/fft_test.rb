require_relative "test_helper"

class FFTTest < Minitest::Test
  def test_fft
    t = Torch.arange(4)
    assert_tensor [6, -2+2i, -2, -2-2i], Torch::FFT.fft(t)
  end

  def test_ifft
    t = Torch.tensor([6, -2+2i, -2, -2-2i])
    assert_tensor [0, 1, 2, 3], Torch::FFT.ifft(t)
  end

  def test_ifft2
    _x = Torch.rand(10, 10, dtype: :complex64)
    # TODO fix
    # Torch::FFT.ifft2(x)
  end

  def test_fftn
    _x = Torch.rand(10, 10, dtype: :complex64)
    # TODO fix
    # Torch::FFT.fftn(x)
  end

  def test_ifftn
    _x = Torch.rand(10, 10, dtype: :complex64)
    # TODO fix
    # Torch::FFT.ifftn(x)
  end

  def test_rfft
    t = Torch.arange(4)
    assert_tensor [6, -2+2i, -2], Torch::FFT.rfft(t)
  end

  def test_irfft
    t = Torch.arange(5)
    t2 = Torch::FFT.rfft(t)
    assert_tensor [0.6250, 1.4045, 3.1250, 4.8455], Torch::FFT.irfft(t2)
    assert_tensor [0, 1, 2, 3, 4], Torch::FFT.irfft(t2, t.numel)
  end

  def test_rfft2
    _t = Torch.rand(10, 10)
    # TODO fix
    # rfft2 = Torch::FFT.rfft2(t)
    # assert_equal [6, 10], rfft2.size
  end

  def test_irfft2
    # TODO
  end

  def test_rfftn
    # TODO
  end

  def test_irfftn
    # TODO
  end

  def test_hfft
    # TODO
  end

  def test_ihfft
    t = Torch.arange(5)
    assert_tensor [2, -0.5-0.6882i, -0.5-0.1625i], Torch::FFT.ihfft(t)
  end

  def test_fftfreq
    # TODO
  end

  def test_rfftfreq
    # TODO
  end

  def test_fftshift
    _f = Torch::FFT.fftfreq(4)
    # TODO fix
    # assert_tensor [-0.5, -0.25,  0,  0.25], Torch::FFT.fftshift(f)
  end

  def test_ifftshift
    assert_tensor [0, 0.2, 0.4, -0.4, -0.2], Torch::FFT.fftfreq(5)
  end
end
