module Torch
  module Utils
    module Data
      class Subset < Dataset
        def initialize(dataset, indices)
          @dataset = dataset
          @indices = indices
        end

        def [](idx)
          @dataset[@indices[idx]]
        end

        def length
          @indices.length
        end
        alias_method :size, :length

        def to_a
          @indices.map { |i| @dataset[i] }
        end
      end
    end
  end
end
