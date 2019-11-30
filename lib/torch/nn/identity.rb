module Torch
  module NN
    class Identity < Module
      def initialize(*args, **options)
        super()
      end

      def forward(input)
        input
      end
    end
  end
end
