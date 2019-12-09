module Torch
  module NN
    class ConstantPad1d < ConstantPadNd
      def initialize(padding, value)
        super(value)
        @padding = _pair(padding)
      end
    end
  end
end
