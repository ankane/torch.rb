module Torch
  module NN
    class LayerNorm < Module
      def initialize(normalized_shape, eps: 1e-5, elementwise_affine: true)
        super()
        @normalized_shape = Array(normalized_shape)
        @eps = eps
        @elementwise_affine = elementwise_affine
        if @elementwise_affine
          @weight = Parameter.new(Torch::Tensor.new(*normalized_shape))
          @bias = Parameter.new(Torch::Tensor.new(*normalized_shape))
        else
          register_parameter("weight", nil)
          register_parameter("bias", nil)
        end
        reset_parameters
      end

      def reset_parameters
        if @elementwise_affine
          Init.ones!(@weight)
          Init.zeros!(@bias)
        end
      end

      def forward(input)
        F.layer_norm(input, @normalized_shape, weight: @weight, bias: @bias, eps: @eps)
      end

      def extra_inspect
        format("%{normalized_shape}, eps: %{eps}, elementwise_affine: %{elementwise_affine}", **dict)
      end
    end
  end
end
