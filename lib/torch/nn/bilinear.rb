module Torch
  module NN
    class Bilinear < Module
      def initialize(in1_features, in2_features, out_features, bias: true)
        super()

        @in1_features = in1_features
        @in2_features = in2_features
        @out_features = out_features
        @weight = Parameter.new(Torch::Tensor.new(out_features, in1_features, in2_features))

        if bias
          @bias = Parameter.new(Torch::Tensor.new(out_features))
        else
          raise NotImplementedYet
        end

        reset_parameters
      end

      def reset_parameters
        bound = 1 / Math.sqrt(@weight.size(1))
        Init.uniform!(@weight, -bound, bound)
        if @bias
          Init.uniform!(@bias, -bound, bound)
        end
      end

      def forward(input1, input2)
        F.bilinear(input1, input2, @weight, @bias)
      end

      def extra_inspect
        format("in1_features: %s, in2_features: %s, out_features: %s, bias: %s", @in1_features, @in2_features, @out_features, !@bias.nil?)
      end
    end
  end
end
