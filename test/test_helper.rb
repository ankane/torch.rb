require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"
require "numo/narray"

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
end
