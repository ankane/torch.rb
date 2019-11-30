module Torch
  module NN
    class Conv2d < ConvNd
      attr_reader :bias, :weight

      def initialize(in_channels, out_channels, kernel_size, stride: 1, padding: 0, dilation: 1, groups: 1, bias: true, padding_mode: "zeros")
        kernel_size = pair(kernel_size)
        stride = pair(stride)
        padding = pair(padding)
        dilation = pair(dilation)
        super(in_channels, out_channels, kernel_size, stride, padding, dilation, false, pair(0), groups, bias, padding_mode)
      end

      def forward(input)
        if @padding_mode == "circular"
          raise NotImplementedError
        end
        F.conv2d(input, @weight, @bias, stride: @stride, padding: @padding, dilation: @dilation, groups: @groups)
      end

      # TODO add more parameters
      def extra_inspect
        format("%s, %s, kernel_size: %s, stride: %s", [@in_channels, @out_channels, @kernel_size, @stride])
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
