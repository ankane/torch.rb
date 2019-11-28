module Torch
  module Optim
    module LRScheduler
      class StepLR
        def initialize(optimizer, step_size:, gamma: 0.1)
          @optimizer = optimizer
          @step_size = step_size
          @gamma = gamma
        end
      end
    end
  end
end
