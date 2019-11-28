module Torch
  module NN
    class DropoutNd < Module
      def initialize(p: 0.5, inplace: false)
        super()
        @p = p
        @inplace = inplace
      end

      def inspect
        "#{self.class.name.split("::").last}(p: #{@p.inspect}, inplace: #{@inplace.inspect})"
      end
    end
  end
end
