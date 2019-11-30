module Torch
  module NN
    class LogSoftmax < Module
      def initialize(dim: nil)
        super()
        @dim = dim
      end

      def forward(input)
        F.log_softmax(input, @dim)
      end
    end
  end
end
