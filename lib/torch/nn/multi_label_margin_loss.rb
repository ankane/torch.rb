module Torch
  module NN
    class MultiLabelMarginLoss < Loss
      def initialize(reduction: "mean")
        super(reduction)
      end

      def forward(input, target)
        F.multilabel_margin_loss(input, target, reduction: @reduction)
      end
    end
  end
end
