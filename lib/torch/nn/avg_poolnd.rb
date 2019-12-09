module Torch
  module NN
    class AvgPoolNd < Module
      def extra_inspect
        format("kernel_size: %s, stride: %s, padding: %s", @kernel_size, @stride, @padding)
      end
    end
  end
end
