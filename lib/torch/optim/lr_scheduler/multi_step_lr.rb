module Torch
  module Optim
    module LRScheduler
      class MultiStepLR < LRScheduler
        def initialize(optimizer, milestones, gamma: 0.1, last_epoch: -1)
          @milestones = milestones.map.with_index.to_h { |v, i| [v, i + 1] }
          @gamma = gamma
          super(optimizer, last_epoch)
        end

        def get_lr
          if !@milestones.include?(@last_epoch)
            @optimizer.param_groups.map { |group| group[:lr] }
          else
            @optimizer.param_groups.map do |group|
              group[:lr] * @gamma ** @milestones[@last_epoch]
            end
          end
        end
      end
    end
  end
end
