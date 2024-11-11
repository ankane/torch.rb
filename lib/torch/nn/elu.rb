module Torch
  module NN
    class ELU < Module
      def initialize(alpha: 1, inplace: false)
        super()
        @alpha = alpha
        @inplace = inplace
      end

      def forward(input)
        F.elu(input, alpha: @alpha, inplace: @inplace)
      end

      def extra_inspect
        inplace_str = @inplace ? ", inplace: true" : ""
        format("alpha: %s", @alpha) + inplace_str
      end
    end
  end
end
