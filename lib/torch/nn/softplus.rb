module Torch
  module NN
    class Softplus < Module
      def initialize(beta: 1, threshold: 20)
        super()
        @beta = beta
        @threshold = threshold
      end

      def forward(input)
        F.softplus(input, beta: @beta, threshold: @threshold)
      end

      def extra_inspect
        format("beta: %s, threshold: %s", [@beta, @threshold])
      end
    end
  end
end
