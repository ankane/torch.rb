module Torch
  module NN
    class NLLLoss < Module
      def forward(input, target)
        F.nll_loss(input, target)
      end
    end
  end
end
