module Torch
  module NN
    class CosineEmbeddingLoss < Loss
      def initialize(margin: 0, reduction: "mean")
        super(reduction)
        @margin = margin
      end

      def forward(input1, input2, target)
        F.cosine_embedding_loss(input1, input2, target, margin: @margin, reduction: @reduction)
      end
    end
  end
end
