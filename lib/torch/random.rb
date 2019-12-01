module Torch
  module Random
    class << self
      # not available through LibTorch
      def initial_seed
        raise NotImplementedYet
      end
    end
  end
end
