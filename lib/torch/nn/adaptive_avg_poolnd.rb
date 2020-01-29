module Torch
  module NN
    class AdaptiveAvgPoolNd < Module
      def initialize(output_size)
        super()
        @output_size  = output_size
      end

      def extra_inspect
        format("output_size: %s", @output_size)
      end
    end
  end
end
