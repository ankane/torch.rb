module Torch
  module NN
    class MaxPoolNd < Module
      def initialize(kernel_size, stride: nil, padding: 0, dilation: 1, return_indices: false, ceil_mode: false)
        super()
        @kernel_size = kernel_size
        @stride = stride || kernel_size
        @padding = padding
        @dilation = dilation
        @return_indices = return_indices
        @ceil_mode = ceil_mode
      end

      def extra_inspect
        s = "kernel_size: %{kernel_size}, stride: %{stride}, padding: %{padding}, dilation: %{dilation}, ceil_mode: %{ceil_mode}"
        format(s, **dict)
      end
    end
  end
end
