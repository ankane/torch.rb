module Torch
  module NN
    class ConstantPad2d < ConstantPadNd
      def initialize(padding, value)
        super(value)
        @padding = _quadrupal(padding)
      end
    end
  end
end
