module Torch
  module NN
    class CosineSimilarity < Module
      def initialize(dim: 1, eps: 1e-8)
        super()
        @dim = dim
        @eps = eps
      end

      def forward(x1, x2)
        F.cosine_similarity(x1, x2, dim: @dim, eps: @eps)
      end
    end
  end
end
