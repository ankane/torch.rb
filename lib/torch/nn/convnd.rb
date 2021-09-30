module Torch
  module NN
    class ConvNd < Module
      attr_reader :in_channels, :out_channels, :kernel_size, :stride, :padding, :dilation, :transposed, :output_paddding, :groups, :padding_mode
      
      def initialize(in_channels, out_channels, kernel_size, stride, padding, dilation, transposed, output_padding, groups, bias, padding_mode)
        super()
        raise ArgumentError, "in_channels must be divisible by groups" if in_channels % groups != 0
        raise ArgumentError, "out_channels must be divisible by groups" if out_channels % groups != 0
        @in_channels = in_channels
        @out_channels = out_channels
        @kernel_size = kernel_size
        @stride = stride
        @padding = padding
        @dilation = dilation
        @transposed = transposed
        @output_padding = output_padding
        @groups = groups
        @padding_mode = padding_mode
        if transposed
          @weight = Parameter.new(Tensor.new(in_channels, out_channels / groups, *kernel_size))
        else
          @weight = Parameter.new(Tensor.new(out_channels, in_channels / groups, *kernel_size))
        end
        if bias
          @bias = Parameter.new(Tensor.new(out_channels))
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
    end
  end
end
