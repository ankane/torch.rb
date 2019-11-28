module Torch
  module NN
    class Dropout < DropoutNd
      def forward(input)
        F.dropout(input, p: @p, training: @training, inplace: @inplace)
      end
    end
  end
end
