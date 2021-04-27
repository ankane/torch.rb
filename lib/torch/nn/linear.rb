module Torch
  module NN
    class Linear < Module
      attr_reader :in_features, :out_features

      def initialize(in_features, out_features, bias: true)
        super()
        @in_features = in_features
        @out_features = out_features

        @weight = Parameter.new(Tensor.new(out_features, in_features))
        if bias
          @bias = Parameter.new(Tensor.new(out_features))
        else
          register_parameter("bias", nil)
        end

        reset_parameters
      end

      def reset_parameters
        Init.kaiming_uniform!(@weight, a: Math.sqrt(5))
        if @bias
          fan_in, _ = Init._calculate_fan_in_and_fan_out(@weight)
          bound = 1 / Math.sqrt(fan_in)
          Init.uniform!(@bias, a: -bound, b: bound)
        end
      end

      def forward(input)
        F.linear(input, @weight, @bias)
      end

      def extra_inspect
        format("in_features: %s, out_features: %s, bias: %s", @in_features, @out_features, !@bias.nil?)
      end
    end
  end
end
