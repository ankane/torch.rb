module Torch
  module NN
    class Unfold < Module
      def initialize(kernel_size, dilation: 1, padding: 0, stride: 1)
        super()
        @kernel_size = kernel_size
        @dilation = dilation
        @padding = padding
        @stride = stride
      end

      def forward(input)
        F.unfold(input, @kernel_size, dilation: @dilation, padding: @padding, stride: @stride)
      end

      # TODO add extra_inspect
    end
  end
end
