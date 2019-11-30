module Torch
  module NN
    class KLDivLoss < Loss
      def initialize(reduction: "mean")
        super(reduction)
      end

      def forward(input, target)
        F.kl_div(input, target, reduction: @reduction)
      end
    end
  end
end
