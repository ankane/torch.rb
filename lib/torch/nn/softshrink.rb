module Torch
  module NN
    class Softshrink < Module
      def initialize(lambd: 0.5)
        super()
        @lambd = lambd
      end

      def forward(input)
        F.softshrink(input, @lambd)
      end

      def extra_inspect
        @lambd.to_s
      end
    end
  end
end
