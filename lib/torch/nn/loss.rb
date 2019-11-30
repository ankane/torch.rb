module Torch
  module NN
    class Loss < Module
      def initialize(reduction)
        @reduction = reduction
      end
    end
  end
end
