module Torch
  module NN
    class ReplicationPad1d < ReplicationPadNd
      def initialize(padding)
        super()
        @padding = _pair(padding)
      end
    end
  end
end
