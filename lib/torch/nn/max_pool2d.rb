module Torch
  module NN
    class MaxPool2d < MaxPoolNd
      def forward(input)
        F.max_pool2d(input, @kernel_size) # TODO other parameters
      end
    end
  end
end
