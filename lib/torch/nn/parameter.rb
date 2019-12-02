module Torch
  module NN
    class Parameter < Tensor
      def self.new(data = nil, requires_grad: true)
        data = Tensor.new unless data
        Tensor._make_subclass(data, requires_grad)
      end

      def inspect
        "Parameter containing:\n#{super}"
      end
    end
  end
end
