module Torch
  module NN
    class ReplicationPad3d < ReplicationPadNd
      def initialize(padding)
        super()
        @padding = _ntuple(6, padding)
      end
    end
  end
end
