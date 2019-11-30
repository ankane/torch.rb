module Torch
  module NN
    class Sequential < Module
      def initialize(*args)
        super()
        # TODO support hash arg (named modules)
        args.each_with_index do |mod, idx|
          add_module(idx.to_s, mod)
        end
      end

      def forward(input)
        @modules.values.each do |mod|
          input = mod.call(input)
        end
        input
      end
    end
  end
end
