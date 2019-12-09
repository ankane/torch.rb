module Torch
  module NN
    class LPPoolNd < Module
      def initialize(norm_type, kernel_size, stride: nil, ceil_mode: false)
        super()
        @norm_type = norm_type
        @kernel_size = kernel_size
        @stride = stride
        @ceil_mode = ceil_mode
      end

      def extra_inspect
        format("norm_type: %{norm_type}, kernel_size: %{kernel_size}, stride: %{stride}, ceil_mode: %{ceil_mode}",
          norm_type: @norm_type,
          kernel_size: @kernel_size,
          stride: @stride,
          ceil_mode: @ceil_mode
        )
      end
    end
  end
end
