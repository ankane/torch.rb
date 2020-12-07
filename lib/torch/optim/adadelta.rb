# ported from https://github.com/pytorch/pytorch/blob/master/torch/optim/adadelta.py
module Torch
  module Optim
    class Adadelta < Optimizer
      def initialize(params, lr: 1.0, rho: 0.9, eps: 1e-6, weight_decay: 0)
        raise ArgumentError, "Invalid learning rate: #{lr}" if lr < 0
        raise ArgumentError, "Invalid rho value: #{rho}" if rho < 0 || rho > 1
        raise ArgumentError, "Invalid epsilon value: #{eps}" if eps < 0
        raise ArgumentError, "Invalid weight_decay value: #{weight_decay}" if weight_decay < 0

        defaults = {lr: lr, rho: rho, eps: eps, weight_decay: weight_decay}
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
              raise Error, "Adadelta does not support sparse gradients"
            end
            state = @state[p]

            if state.size == 0
              state[:step] = 0
              state[:square_avg] = Torch.zeros_like(p.data)
              state[:acc_delta] = Torch.zeros_like(p.data)
            end

            square_avg, acc_delta = state[:square_avg], state[:acc_delta]
            rho, eps = group[:rho], group[:eps]

            state[:step] += 1

            if group[:weight_decay] != 0
              grad = grad.add(p.data, alpha: group[:weight_decay])
            end

            square_avg.mul!(rho).addcmul!(grad, grad, value: 1 - rho)
            std = square_avg.add(eps).sqrt!
            delta = acc_delta.add(eps).sqrt!.div!(std).mul!(grad)
            p.data.add!(delta, alpha: -group[:lr])
            acc_delta.mul!(rho).addcmul!(delta, delta, value: 1 - rho)
          end
        end

        loss
      end
    end
  end
end
