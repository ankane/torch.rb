module Torch
  module NN
    class MaxPool1d < MaxPoolNd
      def forward(input)
        F.max_pool1d(input, @kernel_size, @stride, @padding, @dilation, @ceil_mode, @return_indices)
      end
    end
  end
end
