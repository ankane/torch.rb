module Torch
  module NN
    class InstanceNorm1d < InstanceNorm
      def _check_input_dim(input)
        if input.dim == 2
          raise ArgumentError,
            "InstanceNorm1d returns 0-filled tensor to 2D tensor." +
            "This is because InstanceNorm1d reshapes inputs to" +
            "(1, N * C, ...) from (N, C,...) and this makes" +
            "variances 0."
        end
        if input.dim != 3
          raise "expected 3D input (got #{input.dim}D input)"
        end
      end
    end
  end
end
