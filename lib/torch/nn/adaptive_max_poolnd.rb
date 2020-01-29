module Torch
  module NN
    class AdaptiveMaxPoolNd < Module
      def initialize(output_size) #, return_indices: false)
        super()
        @output_size  = output_size
        # @return_indices = return_indices
      end

      def extra_inspect
        format("output_size: %s", @output_size)
      end
    end
  end
end
