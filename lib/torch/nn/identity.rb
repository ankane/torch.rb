module Torch
  module NN
    class Identity < Module
      # written this way to support unused arguments
      def initialize(*args, **options)
        super()
      end

      def forward(input)
        input
      end
    end
  end
end
