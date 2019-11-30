module Torch
  module NN
    class Softmax2d < Module
      def forward(input)
        raise ArgumentError, "Softmax2d requires a 4D tensor as input" unless input.dim == 4
        F.softmax(input, dim: 1)
      end
    end
  end
end
