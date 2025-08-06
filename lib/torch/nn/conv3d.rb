module Torch
  module NN
    class Conv3d < ConvNd
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
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(in_channels, out_channels, kernel_size, stride, padding, dilation, false, _triple(0), groups, bias, padding_mode)
      end

      def forward(input)
        if @padding_mode == "circular"
          raise NotImplementedError
        end
        F.conv3d(input, @weight, @bias, @stride, @padding, @dilation, @groups)
      end
    end
  end
end
