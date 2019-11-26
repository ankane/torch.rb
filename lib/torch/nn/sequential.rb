module Torch
  module NN
    class Sequential < Module
      def initialize(*args)
        @modules = {}
        # TODO support hash arg (named modules)
        args.each_with_index do |mod, idx|
          add_module(idx.to_s, mod)
        end
      end

      def add_module(name, mod)
        # TODO add checks
        @modules[name] = mod
      end

      def forward(input)
        @modules.values.each do |mod|
          input = mod.call(input)
        end
        input
      end

      def parameters
        @modules.flat_map { |_, mod| mod.parameters }
      end
    end
  end
end
