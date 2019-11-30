module Torch
  module NN
    class MSELoss < Loss
      def initialize(reduction: "mean")
        super(reduction)
      end

      def forward(input, target)
        F.mse_loss(input, target, reduction: @reduction)
      end
    end
  end
end
