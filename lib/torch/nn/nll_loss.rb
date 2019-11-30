module Torch
  module NN
    class NLLLoss < WeightedLoss
      def initialize(weight: nil, ignore_index: -100, reduction: "mean")
        super(weight, reduction)
        @ignore_index = ignore_index
      end

      def forward(input, target)
        F.nll_loss(input, target, weight: @weight, ignore_index: @ignore_index, reduction: @reduction)
      end
    end
  end
end
