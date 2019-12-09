module Torch
  module NN
    class ReflectionPad2d < ReflectionPadNd
      def initialize(padding)
        super()
        @padding = _quadrupal(padding)
      end
    end
  end
end
