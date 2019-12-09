module Torch
  module NN
    class MaxUnpool3d < MaxUnpoolNd
      def initialize(kernel_size, stride: nil, padding: 0)
        super()
        @kernel_size = _triple(kernel_size)
        @stride = _triple(stride || kernel_size)
        @padding = _triple(padding)
      end

      def forward(input, indices, output_size: nil)
        F.max_unpool3d(input, indices, @kernel_size, @stride, @padding, output_size)
      end
    end
  end
end
