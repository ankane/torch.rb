module Torch
  module NN
    class MultiMarginLoss < WeightedLoss
      def initialize(p: 1, margin: 1.0, weight: nil, reduction: "mean")
        super(weight, reduction)
        raise ArgumentError, "only p == 1 and p == 2 supported" if p != 1 && p != 2
        raise ArgumentError, "weight must be nil or have one dimension" unless weight.nil? || weight.dim == 1
        @p = p
        @margin = margin
      end

      def forward(input, target)
        F.multi_margin_loss(input, target, p: @p, margin: @margin, weight: @weight, reduction: @reduction)
      end
    end
  end
end
