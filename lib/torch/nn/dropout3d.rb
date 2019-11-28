module Torch
  module NN
    class Dropout3d < DropoutNd
      def forward(input)
        F.dropout3d(input, p: @p, training: @training, inplace: @inplace)
      end
    end
  end
end
