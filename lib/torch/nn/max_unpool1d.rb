module Torch
  module NN
    class MaxUnpool1d < MaxUnpoolNd
      def initialize(kernel_size, stride: nil, padding: 0)
        super()
        @kernel_size = _single(kernel_size)
        @stride = _single(stride || kernel_size)
        @padding = _single(padding)
      end

      def forward(input, indices, output_size: nil)
        F.max_unpool1d(input, indices, @kernel_size, stride: @stride, padding: @padding, output_size: output_size)
      end
    end
  end
end
