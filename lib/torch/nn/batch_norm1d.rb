module Torch
  module NN
    class BatchNorm1d < BatchNorm
      def _check_input_dim(input)
        if input.dim != 2 && input.dim != 3
          raise ArgumentError, "expected 2D or 3D input (got #{input.dim}D input)"
        end
      end
    end
  end
end
