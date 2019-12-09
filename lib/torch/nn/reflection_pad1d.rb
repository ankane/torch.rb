module Torch
  module NN
    class ReflectionPad1d < ReflectionPadNd
      def initialize(padding)
        super()
        @padding = _pair(padding)
      end
    end
  end
end
