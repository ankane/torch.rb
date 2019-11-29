module Torch
  module Optim
    module LRScheduler
      class LRScheduler
        def initialize(optimizer, last_epoch)
          @optimizer = optimizer
          if last_epoch == -1
            optimizer.param_groups.each do |group|
              group[:initial_lr] ||= group[:lr]
            end
            last_epoch = 0
          else
            raise NotImplementedYet
          end
          @base_lrs = optimizer.param_groups.map { |group| group[:initial_lr] }
          @last_epoch = last_epoch

          @step_count = 0
          step(last_epoch)
        end

        def step(epoch = nil)
          @step_count += 1
          epoch ||= @last_epoch + 1
          @last_epoch = epoch
          @optimizer.param_groups.zip(get_lr).each do |param_group, lr|
            param_group[:lr] = lr
          end
        end
      end
    end
  end
end
