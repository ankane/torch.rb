module Torch
  module NN
    class FeatureAlphaDropout < DropoutNd
      def forward(input)
        F.feature_alpha_dropout(input, p: @p, training: @training, inplace: @inplace)
      end
    end
  end
end
