module Torch
  module NN
    class AdaptiveMaxPool3d < AdaptiveMaxPoolNd
      def forward(input)
        F.adaptive_max_pool3d(input, @output_size) #, @return_indices)
      end
    end
  end
end
