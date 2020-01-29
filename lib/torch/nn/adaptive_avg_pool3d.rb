module Torch
  module NN
    class AdaptiveAvgPool3d < AdaptiveAvgPoolNd
      def forward(input)
        F.adaptive_avg_pool3d(input, @output_size)
      end
    end
  end
end
