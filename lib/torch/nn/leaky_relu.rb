module Torch
  module NN
    class LeakyReLU < Module
      def initialize(negative_slope: 1e-2, inplace: false)
        super()
        @negative_slope = negative_slope
        @inplace = inplace
      end

      def forward(input)
        F.leaky_relu(input, @negative_slope, inplace: @inplace)
      end

      def extra_inspect
        inplace_str = @inplace ? ", inplace: true" : ""
        format("negative_slope: %s%s", @negative_slope, inplace_str)
      end
    end
  end
end
