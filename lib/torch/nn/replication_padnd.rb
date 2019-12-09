module Torch
  module NN
    class ReplicationPadNd < Module
      def forward(input)
        F.pad(input, @padding, mode: "replicate")
      end

      def extra_inspect
        @padding.inspect
      end
    end
  end
end
