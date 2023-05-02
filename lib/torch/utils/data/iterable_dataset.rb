module Torch
  module Utils
    module Data
      class IterableDataset < Dataset
        include Enumerable

        def each
          raise NotImplementedError
        end
      end
    end
  end
end
