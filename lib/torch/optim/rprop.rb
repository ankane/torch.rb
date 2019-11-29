# ported from https://github.com/pytorch/pytorch/blob/master/torch/optim/rprop.py
module Torch
  module Optim
    class Rprop < Optimizer
      def initialize(params, lr: 1e-2, etas: [0.5, 1.2], step_sizes: [1e-6, 50])
        raise ArgumentError, "Invalid learning rate: #{lr}" if lr < 0
        raise ArgumentError, "Invalid eta values: #{etas[0]}, #{etas[1]}" if etas[0] < 0 || etas[0] >= 1 || etas[1] < 1

        defaults = {lr: lr, etas: etas, step_sizes: step_sizes}
        super(params, defaults)
      end

      def step(closure = nil)
        # TODO implement tensor.new
        raise NotImplementedYet

        loss = nil
        if closure
          loss = closure.call
        end

        @param_groups.each do |group|
          group[:params].each do |p|
            next unless p.grad
            grad = p.grad.data
            if grad.sparse?
              raise Error, "Rprop does not support sparse gradients"
            end
            state = @state[p]

            # State initialization
            if state.size == 0
              state[:step] = 0
              state[:prev] = Torch.zeros_like(p.data)
              state[:step_size] = grad.new.resize_as!(grad).fill!(group[:lr])
            end

            etaminus, etaplus = group[:etas]
            step_size_min, step_size_max = group[:step_sizes]
            step_size = state[:step_size]

            state[:step] += 1

            sign = grad.mul(state[:prev]).sign
            sign[sign.gt(0)] = etaplus
            sign[sign.lt(0)] = etaminus
            sign[sign.eq(0)] = 1

            # update stepsizes with step size updates
            step_size.mul!(sign).clamp!(step_size_min, step_size_max)

            # for dir<0, dfdx=0
            # for dir>=0 dfdx=dfdx
            grad = grad.clone
            grad[sign.eq(etaminus)] = 0

            # update parameters
            p.data.addcmul!(-1, grad.sign, step_size)

            state[:prev].copy!(grad)
          end
        end

        loss
      end
    end
  end
end
