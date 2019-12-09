module Torch
  module NN
    class ConstantPad3d < ConstantPadNd
      def initialize(padding, value)
        super(value)
        @padding = _ntuple(6, padding)
      end
    end
  end
end
