# frozen_string_literal: true

require_relative "test_helper"

require "open3"
require "rbconfig"

class TorchRunTest < Minitest::Test
  def test_standalone_launches_multiple_workers
    script = File.expand_path("support/scripts/show_ranks.rb", __dir__)
    torchrun = File.expand_path("../bin/torchrun", __dir__)
    stdout, stderr, status = Open3.capture3(
      {"TORCHRUN_TEST" => "1"},
      RbConfig.ruby,
      torchrun,
      "--standalone",
      "--nproc-per-node=2",
      script
    )

    assert status.success?, "torchrun failed: #{stderr}"

    lines = stdout.lines.map(&:strip).select { |line| line.start_with?("RANK=") }
    assert_equal 2, lines.size, "expected two worker outputs, got: #{lines.inspect}"
    ranks = lines.map do |line|
      match = line.match(/RANK=(\d+)\s+LOCAL_RANK=(\d+)\s+WORLD_SIZE=(\d+)/)
      raise "unexpected output: #{line}" unless match

      [match[1].to_i, match[2].to_i, match[3].to_i]
    end
    assert_equal [[0, 0, 2], [1, 1, 2]], ranks.sort
  end
end
