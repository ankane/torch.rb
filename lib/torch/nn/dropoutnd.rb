module Torch
  module NN
    class DropoutNd < Module
      def initialize(p: 0.5, inplace: false)
        super()
        @p = p
        @inplace = inplace
      end

      def extra_inspect
        format("p: %s, inplace: %s", [@p, @inplace])
      end
    end
  end
end
