module Torch
  module NN
    class BatchNorm3d < BatchNorm
      def _check_input_dim(input)
        if input.dim != 5
          raise ArgumentError, "expected 5D input (got #{input.dim}D input)"
        end
      end
    end
  end
end
