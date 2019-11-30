module Torch
  module NN
    class PoissonNLLLoss < Loss
      def initialize(log_input: true, full: false, eps: 1e-8, reduction: "mean")
        super(reduction)
        @log_input = log_input
        @full = full
        @eps = eps
      end

      def forward(log_input, target)
        F.poisson_nll_loss(log_input, target, log_input: @log_input, full: @full, eps: @eps, reduction: @reduction)
      end
    end
  end
end
