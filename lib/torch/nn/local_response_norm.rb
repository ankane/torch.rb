module Torch
  module NN
    class LocalResponseNorm < Module
      def initialize(size, alpha: 1e-4, beta: 0.75, k: 1.0)
        super()
        @size = size
        @alpha = alpha
        @beta = beta
        @k = k
      end

      def forward(input)
        F.local_response_norm(input, @size, alpha: @alpha, beta: @beta, k: @k)
      end

      def extra_inspect
        format("%{size}, alpha: %{alpha}, beta: %{beta}, k: %{k}", **dict)
      end
    end
  end
end
