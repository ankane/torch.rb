module Torch
  module NN
    class ReplicationPad2d < ReplicationPadNd
      def initialize(padding)
        super()
        @padding = _quadrupal(padding)
      end
    end
  end
end
