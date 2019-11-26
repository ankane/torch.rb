module Torch
  module NN
    class ReLU < Module
      def initialize #(inplace: false)
        # @inplace = inplace
      end

      def forward(input)
        F.relu(input) #, inplace: @inplace)
      end
    end
  end
end
