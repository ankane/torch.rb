module Torch
  module NN
    class LogSigmoid < Module
      def forward(input)
        F.log_sigmoid(input)
      end
    end
  end
end
