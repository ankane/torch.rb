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
        s = String.new("%{in_channels}, %{out_channels}, kernel_size: %{kernel_size}, stride: %{stride}")
        s += ", padding: %{padding}" if @padding != [0] * @padding.size
        s += ", dilation: %{dilation}" if @dilation != [1] * @dilation.size
        s += ", output_padding: %{output_padding}" if @output_padding != [0] * @output_padding.size
        s += ", groups: %{groups}" if @groups != 1
        s += ", bias: false" unless @bias
        s += ", padding_mode: %{padding_mode}" if @padding_mode != "zeros"
        format(s, **dict)
      end
    end
  end
end
