module Torch
  module NN
    class AlphaDropout < DropoutNd
      def forward(input)
        F.alpha_dropout(input, p: @p, training: @training, inplace: @inplace)
      end
    end
  end
end
