module Torch
  module Utils
    module Data
      class DataLoader
        def initialize(dataset, batch_size: 1)
          @dataset = dataset
          @batch_size = batch_size
        end
      end
    end
  end
end
