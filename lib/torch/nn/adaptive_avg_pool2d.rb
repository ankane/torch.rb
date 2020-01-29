module Torch
  module NN
    class AdaptiveAvgPool2d < AdaptiveAvgPoolNd
      def forward(input)
        F.adaptive_avg_pool2d(input, @output_size)
      end
    end
  end
end
