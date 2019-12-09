module Torch
  module NN
    class GroupNorm < Module
      def initialize(num_groups, num_channels, eps: 1e-5, affine: true)
        super()
        @num_groups = num_groups
        @num_channels = num_channels
        @eps = eps
        @affine = affine
        if @affine
          @weight = Parameter.new(Torch::Tensor.new(num_channels))
          @bias = Parameter.new(Torch::Tensor.new(num_channels))
        else
          register_parameter("weight", nil)
          register_parameter("bias", nil)
        end
        reset_parameters
      end

      def reset_parameters
        if @affine
          Init.ones!(@weight)
          Init.zeros!(@bias)
        end
      end

      def forward(input)
        F.group_norm(input, @num_groups, weight: @weight, bias: @bias, eps: @eps)
      end

      def extra_inspect
        format("%{num_groups}, %{num_channels}, eps: %{eps}, affine: %{affine}", **dict)
      end
    end
  end
end
