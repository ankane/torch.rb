module Torch
  module NN
    class BCELoss < WeightedLoss
      def initialize(weight: nil, reduction: "mean")
        super(weight, reduction)
      end

      def forward(input, target)
        F.binary_cross_entropy(input, target, weight: @weight, reduction: @reduction)
      end
    end
  end
end
