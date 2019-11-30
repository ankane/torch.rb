module Torch
  module NN
    class PairwiseDistance < Module
      def initialize(p: 2.0, eps: 1e-6, keepdim: false)
        super()
        @norm = p
        @eps = eps
        @keepdim = keepdim
      end

      def forward(x1, x2)
        F.pairwise_distance(x1, x2, p: @norm, eps: @eps, keepdim: @keepdim)
      end
    end
  end
end
