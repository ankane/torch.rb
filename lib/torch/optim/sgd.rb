module Torch
  module Optim
    class SGD < Optimizer
      def initialize(params, lr:)
        @params = params
        @lr = lr
      end

      def zero_grad
        @params.each do |param|
          if param.grad
            param.grad.detach!
            param.grad.zero!
          end
        end
      end

      def step
        @params.each do |param|
          next unless param.grad
          d_p = param.grad.data
          # same as param.data.add!(-@lr, d_p)
          param.data.sub!(d_p * @lr)
        end
      end
    end
  end
end
