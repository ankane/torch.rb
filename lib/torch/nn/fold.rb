module Torch
  module NN
    class Fold < Module
      def initialize(output_size, kernel_size, dilation: 1, padding: 0, stride: 1)
        super()
        @output_size = output_size
        @kernel_size = kernel_size
        @dilation = dilation
        @padding = padding
        @stride = stride
      end

      def forward(input)
        F.fold(input, @output_size, @kernel_size, dilation: @dilation, padding: @padding, stride: @stride)
      end

      # TODO add extra_inspect
    end
  end
end
