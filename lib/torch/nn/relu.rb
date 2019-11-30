module Torch
  module NN
    class ReLU < Module
      def initialize(inplace: false)
        super()
        @inplace = inplace
      end

      def forward(input)
        F.relu(input, inplace: @inplace)
      end

      def extra_inspect
        @inplace ? "inplace: true" : ""
      end
    end
  end
end
