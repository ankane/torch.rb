module Torch
  module NN
    class MultiLabelSoftMarginLoss < WeightedLoss
      def initialize(weight: nil, reduction: "mean")
        super(weight, reduction)
      end

      def forward(input, target)
        F.multilabel_soft_margin_loss(input, target, weight: @weight, reduction: @reduction)
      end
    end
  end
end
