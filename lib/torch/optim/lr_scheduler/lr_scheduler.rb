module Torch
  module Optim
    module LRScheduler
      class LRScheduler
        def initialize(optimizer, last_epoch)
          @optimizer = optimizer
          @last_epoch = last_epoch
        end
      end
    end
  end
end
