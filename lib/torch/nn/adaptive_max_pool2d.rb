module Torch
  module NN
    class AdaptiveMaxPool2d < AdaptiveMaxPoolNd
      def forward(input)
        F.adaptive_max_pool2d(input, @output_size) #, @return_indices)
      end
    end
  end
end
