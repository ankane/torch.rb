module Torch
  module NN
    class ZeroPad2d < ConstantPad2d
      def initialize(padding)
        super(padding, 0.0)
      end
    end
  end
end
