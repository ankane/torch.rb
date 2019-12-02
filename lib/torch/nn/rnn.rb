module Torch
  module NN
    class RNN < RNNBase
      def initialize(*args, **options)
        if options.key?(:nonlinearity)
          if options[:nonlinearity] == "tanh"
            mode = "RNN_TANH"
          elsif options[:nonlinearity] == "relu"
            mode = "RNN_RELU"
          else
            raise ArgumentError, "Unknown nonlinearity: #{options[:nonlinearity]}"
          end
          options.delete(:nonlinearity)
        else
          mode = "RNN_TANH"
        end

        super(mode, *args, **options)
      end
    end
  end
end
