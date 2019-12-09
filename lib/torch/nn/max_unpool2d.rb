module Torch
  module NN
    class MaxUnpool2d < MaxUnpoolNd
      def initialize(kernel_size, stride: nil, padding: 0)
        super()
        @kernel_size = _pair(kernel_size)
        @stride = _pair(stride || kernel_size)
        @padding = _pair(padding)
      end

      def forward(input, indices, output_size: nil)
        F.max_unpool2d(input, indices, @kernel_size, @stride, @padding, output_size)
      end
    end
  end
end
