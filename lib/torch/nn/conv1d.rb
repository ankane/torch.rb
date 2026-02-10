module Torch
  module NN
    class Conv1d < ConvNd
      def initialize(
        in_channels,
        out_channels,
        kernel_size,
        stride: 1,
        padding: 0,
        dilation: 1,
        groups: 1,
        bias: true,
        padding_mode: "zeros"
      )
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(in_channels, out_channels, kernel_size, stride, padding, dilation, false, _single(0), groups, bias, padding_mode)
      end

      def forward(input)
        if @padding_mode == "circular"
          raise NotImplementedError
        end
        F.conv1d(input, @weight, @bias, @stride, @padding, @dilation, @groups)
      end
    end
  end
end
