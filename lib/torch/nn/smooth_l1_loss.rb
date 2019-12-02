module Torch
  module NN
    class SmoothL1Loss < Loss
      def initialize(reduction: "mean")
        super(reduction)
      end

      def forward(input, target)
        F.smooth_l1_loss(input, target, reduction: @reduction)
      end
    end
  end
end
