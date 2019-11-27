module Torch
  module Utils
    module Data
      class DataLoader
        def initialize(dataset, batch_size: 1)
          @dataset = dataset
          @batch_size = batch_size
        end

        def each
          size = @dataset.size
          start_index = 0
          while start_index < size
            yield @dataset[start_index...(start_index + @batch_size)]
            start_index += @batch_size
          end
        end
      end
    end
  end
end
