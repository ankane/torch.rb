require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"

# support
require_relative "support/net"

class Minitest::Test
  def assert_elements_in_delta(expected, actual)
    assert_equal expected.size, actual.size
    expected.zip(actual) do |exp, act|
      if exp.finite?
        assert_in_delta exp, act
      else
        assert_equal exp, act
      end
    end
  end

  def assert_tensor(expected, actual, dtype: nil)
    assert_kind_of Torch::Tensor, actual
    assert_equal actual.dtype, dtype if dtype
    if (actual.floating_point? || actual.complex?) && actual.dim < 2
      assert_elements_in_delta expected, actual.to_a
    else
      assert_equal expected, actual.to_a
    end
  end

  def mac?
    RbConfig::CONFIG["host_os"] =~ /darwin/i
  end

  def stress_gc
    previous = GC.stress
    begin
      GC.stress = true
      yield
    ensure
      GC.stress = previous
      GC.start
    end
  end
end
