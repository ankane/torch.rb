module Torch
  module Optim
    module LRScheduler
      class CosineAnnealingLR < LRScheduler
        def initialize(optimizer, t_max, eta_min: 0, last_epoch: -1)
          @t_max = t_max
          @eta_min = eta_min
          super(optimizer, last_epoch)
        end

        def get_lr
          if @last_epoch == 0
            @base_lrs
          elsif (@last_epoch - 1 - @t_max) % (2 * @t_max) == 0
            @base_lrs.zip(@optimizer.param_groups).map do |base_lr, group|
              group[:lr] + (base_lr - @eta_min) * (1 - Math.cos(Math::PI / @t_max)) / 2
            end
          else
            @optimizer.param_groups.map do |group|
              (1 + Math.cos(Math::PI * @last_epoch / @t_max)) /
              (1 + Math.cos(Math::PI * (@last_epoch - 1) / @t_max)) *
              (group[:lr] - @eta_min) + @eta_min
            end
          end
        end
      end
    end
  end
end
