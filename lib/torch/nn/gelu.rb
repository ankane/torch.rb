module Torch
  module NN
    class GELU < Module
      def initialize(approximate: 'none')
        super()
        @approximate = approximate
      end

      def forward(input)
        F.gelu(input, approximate: @approximate)
      end

      def extra_inspect
        "approximate: '%{@approximate}'"
      end
    end
  end
end
