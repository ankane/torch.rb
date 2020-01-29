module Torch
  module NN
    class AdaptiveAvgPool1d < AdaptiveAvgPoolNd
      def forward(input)
        F.adaptive_avg_pool1d(input, @output_size)
      end
    end
  end
end
