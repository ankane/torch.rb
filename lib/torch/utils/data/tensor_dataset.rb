module Torch
  module Utils
    module Data
      class TensorDataset
        def initialize(*tensors)
          @tensors = tensors
        end

        def [](index)
          tensors.map { |t| t[index] }
        end
      end
    end
  end
end
