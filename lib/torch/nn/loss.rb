module Torch
  module NN
    class Loss < Module
      def initialize(reduction)
        super()
        @reduction = reduction
      end
    end
  end
end
