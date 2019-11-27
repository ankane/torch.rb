module Torch
  module NN
    class Parameter < Tensor
      def self.new(data = nil, requires_grad: true)
        data = Tensor.new unless data
        Tensor._make_subclass(data, requires_grad)
      end

      def grad
        if _grad_defined
          _grad
        else
          nil
        end
      end
    end
  end
end
