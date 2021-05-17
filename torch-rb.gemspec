require_relative "lib/torch/version"

Gem::Specification.new do |spec|
  spec.name          = "torch-rb"
  spec.version       = Torch::VERSION
  spec.summary       = "Deep learning for Ruby, powered by LibTorch"
  spec.homepage      = "https://github.com/ankane/torch.rb"
  spec.license       = "BSD-3-Clause"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@ankane.org"

  spec.files         = Dir["*.{md,txt}", "{codegen,ext,lib}/**/*"]
  spec.require_path  = "lib"
  spec.extensions    = ["ext/torch/extconf.rb"]

  spec.required_ruby_version = ">= 2.6"

  spec.add_dependency "rice", ">= 4"
end
