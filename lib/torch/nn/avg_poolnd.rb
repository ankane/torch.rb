module Torch
  module NN
    class AvgPoolNd < Module
      def extra_inspect
        format("kernel_size=%s", @kernel_size)
      end
    end
  end
end
