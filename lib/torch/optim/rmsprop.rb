# ported from https://github.com/pytorch/pytorch/blob/master/torch/optim/rmsprop.py
module Torch
  module Optim
    class RMSprop < Optimizer
      def initialize(params, lr: 1e-2, alpha: 0.99, eps: 1e-8, weight_decay: 0, momentum: 0, centered: false)
        raise ArgumentError, "Invalid learning rate: #{lr}" if lr < 0
        raise ArgumentError, "Invalid epsilon value: #{eps}" if eps < 0
        raise ArgumentError, "Invalid momentum value: #{momentum}" if momentum < 0
        raise ArgumentError, "Invalid weight_decay value: #{weight_decay}" if weight_decay < 0
        raise ArgumentError, "Invalid momentum alpha: #{alpha}" if alpha < 0

        defaults = {lr: lr, momentum: momentum, alpha: alpha, eps: eps, centered: centered, weight_decay: weight_decay}
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
              raise Error, "RMSprop does not support sparse gradients"
            end
            state = @state[p]

            # State initialization
            if state.size == 0
              state[:step] = 0
              state[:square_avg] = Torch.zeros_like(p.data)
              if group[:momentum] > 0
                state[:momentum_buffer] = Torch.zeros_like(p.data)
              end
              if group[:centered]
                state[:grad_avg] = Torch.zeros_like(p.data)
              end
            end

            square_avg = state[:square_avg]
            alpha = group[:alpha]

            state[:step] += 1

            if group[:weight_decay] != 0
              grad = grad.add(p.data, alpha: group[:weight_decay])
            end

            square_avg.mul!(alpha).addcmul!(1 - alpha, grad, grad)

            if group[:centered]
              grad_avg = state[:grad_avg]
              grad_avg.mul!(alpha).add!(grad, alpha: 1 - alpha)
              avg = square_avg.addcmul(grad_avg, grad_avg, value: -1).sqrt!.add!(group[:eps])
            else
              avg = square_avg.sqrt.add!(group[:eps])
            end

            if group[:momentum] > 0
              buf = state[:momentum_buffer]
              buf.mul!(group[:momentum]).addcdiv!(1, grad, avg)
              p.data.add!(buf, alpha: -group[:lr])
            else
              p.data.addcdiv!(-group[:lr], grad, avg)
            end
          end
        end

        loss
      end
    end
  end
end
