module Torch
  module Utils
    module Data
      class DataLoader
        include Enumerable

        attr_reader :dataset

        def initialize(dataset, batch_size: 1)
          @dataset = dataset
          @batch_size = batch_size
        end

        def each
          size.times do |i|
            start_index = i * @batch_size
            yield @dataset[start_index...(start_index + @batch_size)]
          end
        end

        def size
          (@dataset.size / @batch_size.to_f).ceil
        end
      end
    end
  end
end
