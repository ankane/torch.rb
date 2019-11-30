module Torch
  module NN
    class PReLU < Module
      def initialize(num_parameters: 1, init: 0.25)
        @num_parameters = num_parameters
        super()
        @weight = Parameter.new(Torch::Tensor.new(num_parameters).fill!(init))
      end

      def forward(input)
        F.prelu(input, @weight)
      end

      def extra_inspect
        format("num_parameters: %s", @num_parameters)
      end
    end
  end
end
