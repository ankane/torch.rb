require "bundler/gem_tasks"
require "rake/testtask"
require "rake/extensiontask"

task default: :test
Rake::TestTask.new do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
  t.warning = false
end

Rake::ExtensionTask.new("torch") do |ext|
  ext.name = "ext"
  ext.lib_dir = "lib/torch"
end

# include ext in local installs but not releases
task :remove_ext do
  path = "lib/torch/ext.bundle"
  File.unlink(path) if File.exist?(path)
end

Rake::Task["release:guard_clean"].enhance [:remove_ext]

namespace :benchmark do
  task :numo do
    require "benchmark"
    require "numo/narray"
    require "torch-rb"

    x = Numo::SFloat.new(60000, 28, 28).seq
    p Benchmark.realtime { Torch.from_numo(x) }
  end
end
