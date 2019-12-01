module Torch
  module NN
    class MaxPoolNd < Module
      def initialize(kernel_size) #, stride: nil, padding: 0, dilation: 1, return_indices: false, ceil_mode: false)
        super()
        @kernel_size = kernel_size
        # @stride = stride || kernel_size
        # @padding = padding
        # @dilation = dilation
        # @return_indices = return_indices
        # @ceil_mode = ceil_mode
      end

      def extra_inspect
        format("kernel_size: %s", @kernel_size)
      end
    end
  end
end
