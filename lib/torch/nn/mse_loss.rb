module Torch
  module NN
    class MSELoss < Module
      def initialize(reduction: "mean")
        @reduction = reduction
      end

      def forward(input, target)
        F.mse_loss(input, target, reduction: @reduction)
      end
    end
  end
end
