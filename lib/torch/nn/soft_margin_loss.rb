module Torch
  module NN
    class SoftMarginLoss < Loss
      def initialize(reduction: "mean")
        super(reduction)
      end

      def forward(input, target)
        F.soft_margin_loss(input, target, reduction: @reduction)
      end
    end
  end
end
