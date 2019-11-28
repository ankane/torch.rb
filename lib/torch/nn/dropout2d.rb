module Torch
  module NN
    class Dropout2d < DropoutNd
      def forward(input)
        F.dropout2d(input, p: @p, training: @training, inplace: @inplace)
      end
    end
  end
end
