module Torch
  module NN
    class Softmin < Module
      def initialize(dim: nil)
        super()
        @dim = dim
      end

      def forward(input)
        F.softmin(input, dim: @dim)
      end
    end
  end
end
