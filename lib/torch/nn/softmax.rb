module Torch
  module NN
    class Softmax < Module
      def initialize(dim: nil)
        super()
        @dim = dim
      end

      def forward(input)
        F.softmax(input, dim: @dim)
      end

      def extra_inspect
        format("dim: %s", [@dim])
      end
    end
  end
end
