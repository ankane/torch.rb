# frozen_string_literal: true

$stdout.sync = true
rank = ENV.fetch("RANK", "unknown")
local_rank = ENV.fetch("LOCAL_RANK", "unknown")
world_size = ENV.fetch("WORLD_SIZE", "unknown")
puts "RANK=#{rank} LOCAL_RANK=#{local_rank} WORLD_SIZE=#{world_size}"
