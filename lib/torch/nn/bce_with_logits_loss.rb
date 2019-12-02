module Torch
  module NN
    class BCEWithLogitsLoss < Loss
      def initialize(weight: nil, reduction: "mean", pos_weight: nil)
        super(reduction)
        register_buffer("weight", weight)
        register_buffer("pos_weight", pos_weight)
      end

      def forward(input, target)
        F.binary_cross_entropy_with_logits(input, target, weight: weight, pos_weight: pos_weight, reduction: @reduction)
      end
    end
  end
end
