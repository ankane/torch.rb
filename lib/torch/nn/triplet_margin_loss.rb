module Torch
  module NN
    class TripletMarginLoss < Loss
      def initialize(margin: 1.0, p: 2.0, eps: 1e-6, swap: false, reduction: "mean")
        super(reduction)
        @margin = margin
        @p = p
        @eps = eps
        @swap = swap
      end

      def forward(anchor, positive, negative)
        F.triplet_margin_loss(anchor, positive, negative, margin: @margin, p: @p,
                              eps: @eps, swap: @swap, reduction: @reduction)
      end
    end
  end
end
