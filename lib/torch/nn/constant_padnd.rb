module Torch
  module NN
    class ConstantPadNd < Module
      def initialize(value)
        super()
        @value = value
      end

      def forward(input)
        F.pad(input, @padding, mode: "constant", value: @value)
      end

      def extra_inspect
        format("padding: %s, value: %s", @padding, @value)
      end
    end
  end
end
