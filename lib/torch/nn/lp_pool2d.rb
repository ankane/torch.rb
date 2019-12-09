module Torch
  module NN
    class LPPool2d < LPPoolNd
      def forward(input)
        F.lp_pool2d(input, @norm_type.to_f, @kernel_size, @stride, @ceil_mode)
      end
    end
  end
end
