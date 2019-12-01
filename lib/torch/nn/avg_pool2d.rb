module Torch
  module NN
    class AvgPool2d < AvgPoolNd
      def initialize(kernel_size)
        super()
        @kernel_size = kernel_size
      end

      def forward(input)
        F.avg_pool2d(input, @kernel_size)
      end
    end
  end
end
