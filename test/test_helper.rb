spawn_worker = ENV["TORCH_DISTRIBUTED_SPAWNED"] == "1"

# Spawned distributed workers shouldn't try to load minitest plugins from the
# parent test environment.
ENV["MT_NO_PLUGINS"] = "1" if spawn_worker

require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"

if spawn_worker
  module TorchDistributedSpawnTest
    module QuietSummaryReporter
      def start # :nodoc:
        Minitest::StatisticsReporter.instance_method(:start).bind(self).call
        self.sync = io.respond_to?(:"sync=")
        self.old_sync, io.sync = io.sync, true if self.sync
      end

      def report # :nodoc:
        super
      ensure
        io.sync = self.old_sync if self.sync
      end
    end
  end

  Minitest::SummaryReporter.prepend(TorchDistributedSpawnTest::QuietSummaryReporter)
end

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
