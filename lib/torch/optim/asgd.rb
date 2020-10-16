# ported from https://github.com/pytorch/pytorch/blob/master/torch/optim/asgd.py
module Torch
  module Optim
    class ASGD < Optimizer
      def initialize(params, lr: 1e-2, lambd: 1e-4, alpha: 0.75, t0: 1e6, weight_decay: 0)
        raise ArgumentError, "Invalid learning rate: #{lr}" if lr < 0
        raise ArgumentError, "Invalid weight_decay value: #{weight_decay}" if weight_decay < 0

        defaults = {lr: lr, lambd: lambd, alpha: alpha, t0: t0, weight_decay: weight_decay}
        super(params, defaults)
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
            if grad.sparse?
              raise Error, "ASGD does not support sparse gradients"
            end
            state = @state[p]

            # State initialization
            if state.size == 0
              state[:step] = 0
              state[:eta] = group[:lr]
              state[:mu] = 1
              state[:ax] = Torch.zeros_like(p.data)
            end

            state[:step] += 1

            if group[:weight_decay] != 0
              grad = grad.add(p.data, alpha: group[:weight_decay])
            end

            # decay term
            p.data.mul!(1 - group[:lambd] * state[:eta])

            # update parameter
            p.data.add!(grad, alpha: -state[:eta])

            # averaging
            if state[:mu] != 1
              state[:ax].add!(p.data.sub(state[:ax]).mul(state[:mu]))
            else
              state[:ax].copy!(p.data)
            end

            # update eta and mu
            state[:eta] = (group[:lr] / ((1 + group[:lambd] * group[:lr] * state[:step]) ** group[:alpha]))
            state[:mu] = 1 / [1, state[:step] - group[:t0]].max
          end
        end

        loss
      end
    end
  end
end
