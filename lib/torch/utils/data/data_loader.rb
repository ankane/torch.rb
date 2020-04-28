module Torch
  module Utils
    module Data
      class DataLoader
        include Enumerable

        attr_reader :dataset

        def initialize(dataset, batch_size: 1, shuffle: false)
          @dataset = dataset
          @batch_size = batch_size
          @shuffle = shuffle
        end

        def each
          # try to keep the random number generator in sync with Python
          # this makes it easy to compare results
          base_seed = Torch.empty([], dtype: :int64).random!.item

          indexes =
            if @shuffle
              Torch.randperm(@dataset.size).to_a
            else
              @dataset.size.times
            end

          indexes.each_slice(@batch_size) do |idx|
            batch = idx.map { |i| @dataset[i] }
            yield collate(batch)
          end
        end

        def size
          (@dataset.size / @batch_size.to_f).ceil
        end

        private

        def collate(batch)
          elem = batch[0]
          case elem
          when Tensor
            Torch.stack(batch, 0)
          when Integer
            Torch.tensor(batch)
          when Array
            batch.transpose.map { |v| collate(v) }
          else
            raise NotImpelmentYet
          end
        end
      end
    end
  end
end
