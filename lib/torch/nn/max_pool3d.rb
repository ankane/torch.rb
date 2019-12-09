module Torch
  module NN
    class MaxPool3d < MaxPoolNd
      def forward(input)
        raise NotImplemntedYet if @return_indices
        F.max_pool3d(input, @kernel_size, @stride, @padding, @dilation, @ceil_mode) #, @return_indices)
      end
    end
  end
end
