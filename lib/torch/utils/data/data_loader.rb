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
          max_size = @dataset.size
          size.times do |i|
            start_index = i * @batch_size
            end_index = [start_index + @batch_size, max_size].min
            batch = (end_index - start_index).times.map { |j| @dataset[start_index + j] }
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
