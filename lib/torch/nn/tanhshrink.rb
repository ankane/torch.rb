module Torch
  module NN
    class Tanhshrink < Module
      def forward(input)
        F.tanhshrink(input)
      end
    end
  end
end
