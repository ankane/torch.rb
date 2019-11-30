module Torch
  module NN
    class CTCLoss < Loss
      def initialize(blank: 0, reduction: "mean", zero_infinity: false)
        super(reduction)
        @blank = blank
        @zero_infinity = zero_infinity
      end

      def forward(log_probs, targets, input_lengths, target_lengths)
        F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank: @blank, reduction: @reduction, zero_infinity: @zero_infinity)
      end
    end
  end
end
