module Torch
  module Utils
    module Data
      class TensorDataset < Dataset
        def initialize(*tensors)
          unless tensors.all? { |t| t.size(0) == tensors[0].size(0) }
            raise Error, "Tensors must all have same dim 0 size"
          end
          @tensors = tensors
        end

        def [](index)
          @tensors.map { |t| t[index] }
        end

        def size
          @tensors[0].size(0)
        end
        alias_method :length, :size
        alias_method :count, :size
      end
    end
  end
end
