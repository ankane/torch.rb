module Torch
  module Utils
    module Data
      class TensorDataset
        def initialize(*tensors)
          @tensors = tensors
        end

        def [](index)
          @tensors.map { |t| t[index] }
        end

        def size
          @tensors[0].size(0)
        end
      end
    end
  end
end
