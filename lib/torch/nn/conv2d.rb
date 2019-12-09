module Torch
  module NN
    class Conv2d < ConvNd
      def initialize(in_channels, out_channels, kernel_size, stride: 1,
        padding: 0, dilation: 1, groups: 1, bias: true, padding_mode: "zeros")

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(in_channels, out_channels, kernel_size, stride, padding, dilation, false, _pair(0), groups, bias, padding_mode)
      end

      def forward(input)
        if @padding_mode == "circular"
          raise NotImplementedError
        end
        F.conv2d(input, @weight, @bias, @stride, @padding, @dilation, @groups)
      end

      # TODO add more parameters
      def extra_inspect
        format("%s, %s, kernel_size: %s, stride: %s", @in_channels, @out_channels, @kernel_size, @stride)
      end
    end
  end
end
