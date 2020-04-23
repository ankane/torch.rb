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
  if Rake.application.top_level_tasks.include?("release")
    Dir["lib/torch/ext.bundle", "ext/torch/*_functions.{cpp,hpp}"].each do |path|
      File.unlink(path) if File.exist?(path)
    end
  end
end

Rake::Task["build"].enhance [:remove_ext]

namespace :generate do
  desc "Generate C++ functions"
  task :functions do
    require_relative "lib/torch/native/generator"
    Torch::Native::Generator.generate_cpp_functions
  end
end

namespace :benchmark do
  desc "Benchmark Numo"
  task :numo do
    require "benchmark"
    require "numo/narray"
    require "torch-rb"

    x = Numo::SFloat.new(60000, 28, 28).seq
    t = nil
    p Benchmark.realtime { t = Torch.from_numo(x) }
    p Benchmark.realtime { t.numo }
  end
end
