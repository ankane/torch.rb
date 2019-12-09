module Torch
  module NN
    class Softsign < Module
      def forward(input)
        F.softsign(input)
      end
    end
  end
end
