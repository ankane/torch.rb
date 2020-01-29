module Torch
  module NN
    class AdaptiveMaxPool1d < AdaptiveMaxPoolNd
      def forward(input)
        F.adaptive_max_pool1d(input, @output_size) #, @return_indices)
      end
    end
  end
end
