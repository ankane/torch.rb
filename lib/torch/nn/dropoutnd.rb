module Torch
  module NN
    class DropoutNd < Module
      def initialize(p = 0.5, inplace: false)
        super
        @p = p
        @inplace = inplace
      end

      # TODO inspect method
    end
  end
end
