module Torch
  module NN
    class CrossEntropyLoss < WeightedLoss
      def initialize(weight: nil, ignore_index: -100, reduction: "mean")
        super(weight, reduction)
        @ignore_index = ignore_index
      end

      def forward(input, target)
        F.cross_entropy(input, target, weight: @weight, ignore_index: @ignore_index, reduction: @reduction)
      end
    end
  end
end
