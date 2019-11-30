module Torch
  module NN
    class Sigmoid < Module
      def forward(input)
        Torch.sigmoid(input)
      end
    end
  end
end
