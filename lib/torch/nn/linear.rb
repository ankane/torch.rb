module Torch
  module NN
    class Linear < Module
      attr_reader :bias, :weight

      def initialize(in_features, out_features, bias: true)
        @in_features = in_features
        @out_features = out_features

        @weight = Parameter.new(Tensor.new(out_features, in_features))
        if bias
          @bias = Parameter.new(Tensor.new(out_features))
        end

        reset_parameters
      end

      def call(input)
        F.linear(input, @weight, @bias)
      end

      def reset_parameters
        Init.kaiming_uniform_(@weight, Math.sqrt(5))
        if @bias
          fan_in, _ = Init.calculate_fan_in_and_fan_out(@weight)
          bound = 1 / Math.sqrt(fan_in)
          Init.uniform_(@bias, -bound, bound)
        end
      end

      def inspect
        "Linear(in_features: #{@in_features.inspect}, out_features: #{@out_features.inspect}, bias: #{(!@bias.nil?).inspect})"
      end
    end
  end
end
