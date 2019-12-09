module Torch
  module NN
    class AvgPool2d < AvgPoolNd
      def initialize(kernel_size, stride: nil, padding: 0, ceil_mode: false, count_include_pad: true, divisor_override: nil)
        super()
        @kernel_size = kernel_size
        @stride = stride || kernel_size
        @padding = padding
        @ceil_mode = ceil_mode
        @count_include_pad = count_include_pad
        @divisor_override = divisor_override
      end

      def forward(input)
        F.avg_pool2d(input, @kernel_size, @stride, @padding, @ceil_mode, @count_include_pad, @divisor_override)
      end
    end
  end
end
