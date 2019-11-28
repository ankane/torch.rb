module Torch
  module Optim
    class Adadelta < Optimizer
      def initialize(params, lr: 1.0, rho: 0.9, eps: 1e-6, weight_decay: 0)
        super()
        @params = params
        @lr = lr
        @rho = rho
        @eps = eps
        @weight_decay = weight_decay
      end

      def step
        @params.each do |param|
          next unless param.grad
          grad = param.grad.data

          if grad.sparse?
            raise Error, "Adadelta does not support sparse gradients"
          end

          state = @state[param] || {}

          if state.size == 0
            state[:step] = 0
            state[:square_avg] = Torch.zeros_like(param.data)
            state[:acc_delta] = Torch.zeros_like(param.data)
          end

          square_avg, acc_delta = state[:square_avg], state[:acc_delta]
          rho, eps = @rho, @eps

          state[:step] += 1

          if @weight_decay != 0
            grad = grad.add(param.data * @weight_decay)
          end

          square_avg.mul!(rho).addcmul!(1 - rho, grad, grad)
          std = square_avg.add(eps).sqrt!
          delta = acc_delta.add(eps).sqrt!.div!(std).mul!(grad)
          param.data.add!(-@lr, delta)
          acc_delta.mul!(rho).addcmul!(1 - rho, delta, delta)
        end
      end
    end
  end
end
