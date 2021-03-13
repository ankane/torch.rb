module Torch
  module Utils
    module Data
      class DataLoader
        include Enumerable

        attr_reader :dataset

        def initialize(dataset, batch_size: 1, shuffle: false, collate_fn: nil)
          @dataset = dataset
          @batch_size = batch_size
          @shuffle = shuffle

          @batch_sampler = nil

          if collate_fn.nil?
            if auto_collation?
              collate_fn = method(:default_collate)
            else
              collate_fn = method(:default_convert)
            end
          end

          @collate_fn = collate_fn
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
            # TODO improve performance
            yield @collate_fn.call(idx.map { |i| @dataset[i] })
          end
        end

        def size
          (@dataset.size / @batch_size.to_f).ceil
        end
        alias_method :length, :size
        alias_method :count, :size

        private

        def default_convert(batch)
          elem = batch[0]
          case elem
          when Tensor
            Torch.stack(batch, 0)
          when Integer
            Torch.tensor(batch)
          when Array
            batch.transpose.map { |v| default_convert(v) }
          else
            batch
          end
        end

        def auto_collation?
          !@batch_sampler.nil?
        end
      end
    end
  end
end
