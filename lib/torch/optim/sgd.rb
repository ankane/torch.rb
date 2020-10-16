# ported from https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
module Torch
  module Optim
    class SGD < Optimizer
      def initialize(params, lr:, momentum: 0, dampening: 0, weight_decay: 0, nesterov: false)
        raise ArgumentError, "Invalid learning rate: #{lr}" if lr < 0.0
        raise ArgumentError, "Invalid momentum value: #{momentum}" if momentum < 0.0
        raise ArgumentError, "Invalid weight_decay value: #{weight_decay}" if weight_decay < 0.0

        defaults = {lr: lr, momentum: momentum, dampening: dampening, weight_decay: weight_decay, nesterov: nesterov}

        if nesterov && (momentum <= 0 || dampening != 0)
          raise ArgumentError, "Nesterov momentum requires a momentum and zero dampening"
        end

        super(params, defaults)
      end

      def step(closure = nil)
        loss = nil
        if closure
          loss = closure.call
        end

        @param_groups.each do |group|
          weight_decay = group[:weight_decay]
          momentum = group[:momentum]
          dampening = group[:dampening]
          nesterov = group[:nesterov]

          group[:params].each do |p|
            next unless p.grad
            d_p = p.grad.data
            if weight_decay != 0
              d_p.add!(p.data, alpha: weight_decay)
            end
            if momentum != 0
              param_state = @state[p]
              if !param_state.key?(:momentum_buffer)
                buf = param_state[:momentum_buffer] = Torch.clone(d_p).detach
              else
                buf = param_state[:momentum_buffer]
                buf.mul!(momentum).add!(d_p, alpha: 1 - dampening)
              end
              if nesterov
                d_p = d_p.add(buf, alpha: momentum)
              else
                d_p = buf
              end
            end

            p.data.add!(d_p, alpha: -group[:lr])
          end
        end

        loss
      end
    end
  end
end
