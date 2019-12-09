module Torch
  module NN
    class ReflectionPadNd < Module
      def forward(input)
        F.pad(input, @padding, mode: "reflect")
      end

      def extra_inspect
        @padding.inspect
      end
    end
  end
end
