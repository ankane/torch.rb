# ported from https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
module Torch
  module Optim
    class Adagrad < Optimizer
      def initialize(params, lr: 1e-2, lr_decay: 0, weight_decay: 0, initial_accumulator_value: 0, eps: 1e-10)
        raise ArgumentError, "Invalid learning rate: #{lr}" if lr < 0
        raise ArgumentError, "Invalid lr_decay value: #{lr_decay}" if lr_decay < 0
        raise ArgumentError, "Invalid initial_accumulator_value value: #{initial_accumulator_value}" if initial_accumulator_value < 0
        raise ArgumentError, "Invalid weight_decay value: #{weight_decay}" if weight_decay < 0
        raise ArgumentError, "Invalid epsilon value: #{eps}" if eps < 0

        defaults = {lr: lr, lr_decay: lr_decay, eps: eps, weight_decay: weight_decay, initial_accumulator_value: initial_accumulator_value}
        super(params, defaults)

        @param_groups.each do |group|
          group[:params].each do |p|
            state = @state[p]
            state[:step] = 0
            state[:sum] = Torch.full_like(p.data, initial_accumulator_value)
          end
        end
      end

      def share_memory
        @param_groups.each do |group|
          group[:params].each do |p|
            state = @state[p]
            state[:sum].share_memory!
          end
        end
      end

      def step(closure = nil)
        loss = nil
        if closure
          loss = closure.call
        end

        @param_groups.each do |group|
          group[:params].each do |p|
            next unless p.grad

            grad = p.grad.data
            state = @state[p]

            state[:step] += 1

            if group[:weight_decay] != 0
              if p.grad.data.sparse?
                raise Error, "weight_decay option is not compatible with sparse gradients"
              end
              grad = grad.add(p.data, alpha: group[:weight_decay])
            end

            clr = group[:lr] / (1 + (state[:step] - 1) * group[:lr_decay])

            if grad.sparse?
              raise NotImplementedYet
            else
              state[:sum].addcmul!(grad, grad, value: 1)
              std = state[:sum].sqrt.add!(group[:eps])
              p.data.addcdiv!(grad, std, value: -clr)
            end
          end
        end

        loss
      end
    end
  end
end
