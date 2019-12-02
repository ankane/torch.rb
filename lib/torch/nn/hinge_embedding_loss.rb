module Torch
  module NN
    class HingeEmbeddingLoss < Loss
      def initialize(margin: 1.0, reduction: "mean")
        super(reduction)
        @margin = margin
      end

      def forward(input, target)
        F.hinge_embedding_loss(input, target, margin: @margin, reduction: @reduction)
      end
    end
  end
end
