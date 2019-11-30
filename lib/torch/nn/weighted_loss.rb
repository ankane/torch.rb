module Torch
  module NN
    class WeightedLoss < Loss
      def initialize(weight, reduction)
        super(reduction)
        register_buffer("weight", weight)
      end
    end
  end
end
