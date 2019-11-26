module Torch
  module NN
    class Conv2d < Module
      attr_reader :bias, :weight

      def initialize(in_channels, out_channels, kernel_size) #, stride: 1, padding: 0, dilation: 1, groups: 1)
        @in_channels = in_channels
        @out_channels = out_channels
        @kernel_size = pair(kernel_size)
        @stride = pair(1)
        # @stride = pair(stride)
        # @padding = pair(padding)
        # @dilation = pair(dilation)

        # TODO divide by groups
        @weight = Parameter.new(Tensor.new(out_channels, in_channels, *@kernel_size))
        @bias = Parameter.new(Tensor.new(out_channels))

        reset_parameters
      end

      def reset_parameters
        Init.kaiming_uniform_(@weight, Math.sqrt(5))
        if @bias
          fan_in, _ = Init.calculate_fan_in_and_fan_out(@weight)
          bound = 1 / Math.sqrt(fan_in)
          Init.uniform_(@bias, -bound, bound)
        end
      end

      def call(input)
        F.conv2d(input, @weight, @bias) # @stride, self.padding, self.dilation, self.groups)
      end

      def inspect
        "Conv2d(#{@in_channels}, #{@out_channels}, kernel_size: #{@kernel_size.inspect}, stride: #{@stride.inspect})"
      end

      private

      def pair(value)
        if value.is_a?(Array)
          value
        else
          [value] * 2
        end
      end
    end
  end
end
