module Torch
  module Optim
    module LRScheduler
      class ExponentialLR < LRScheduler
        def initialize(optimizer, gamma, last_epoch: -1)
          @gamma = gamma
          super(optimizer, last_epoch)
        end

        def get_lr
          if @last_epoch == 0
            @base_lrs
          else
            @optimizer.param_groups.map do |group|
              group[:lr] * @gamma
            end
          end
        end
      end
    end
  end
end
