module Torch
  module NN
    class AvgPool1d < AvgPoolNd
      def initialize(kernel_size, stride: nil, padding: 0, ceil_mode: false, count_include_pad: true)
        super()
        @kernel_size = _single(kernel_size)
        @stride = _single(stride || kernel_size)
        @padding = _single(padding)
        @ceil_mode = ceil_mode
        @count_include_pad = count_include_pad
      end

      def forward(input)
        F.avg_pool1d(input, @kernel_size, @stride, @padding, @ceil_mode, @count_include_pad)
      end
    end
  end
end
