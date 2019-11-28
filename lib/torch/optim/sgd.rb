module Torch
  module Optim
    class SGD < Optimizer
      # TODO support momentum, dampening, weight_decay, nesterov
      def initialize(params, lr:)
        super()
        @params = params
        @lr = lr
      end

      def step
        @params.each do |param|
          next unless param.grad
          d_p = param.grad.data
          param.data.add!(-@lr, d_p)
        end
      end
    end
  end
end
