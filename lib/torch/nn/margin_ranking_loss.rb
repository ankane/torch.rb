module Torch
  module NN
    class MarginRankingLoss < Loss
      def initialize(margin: 1.0, reduction: "mean")
        super(reduction)
        @margin = margin
      end

      def forward(input1, input2, target)
        F.margin_ranking_loss(input1, input2, target, margin: @margin, reduction: @reduction)
      end
    end
  end
end
