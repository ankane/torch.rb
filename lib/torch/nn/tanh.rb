module Torch
  module NN
    class Tanh < Module
      def forward(input)
        Torch.tanh(input)
      end
    end
  end
end
