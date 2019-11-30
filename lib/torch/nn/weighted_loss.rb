module Torch
  module NN
    class WeightedLoss < Loss
      def initialize(weight, reduction)
        super(reduction)
        # TODO
        # register_buffer("weight", weight)
      end
    end
  end
end
