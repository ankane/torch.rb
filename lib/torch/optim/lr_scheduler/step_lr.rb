module Torch
  module Optim
    module LRScheduler
      class StepLR < LRScheduler
        def initialize(optimizer, step_size:, gamma: 0.1, last_epoch: -1)
          @step_size = step_size
          @gamma = gamma
          super(optimizer, last_epoch)
        end

        def step
          raise NotImplementedYet
        end
      end
    end
  end
end
