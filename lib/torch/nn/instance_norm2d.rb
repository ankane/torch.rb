module Torch
  module NN
    class InstanceNorm2d < InstanceNorm
      def _check_input_dim(input)
        if input.dim != 4
          raise ArgumentError, "expected 4D input (got #{input.dim}D input)"
        end
      end
    end
  end
end
