module Torch
  module Utils
    module Data
      class << self
        def random_split(dataset, lengths)
          if lengths.sum != dataset.length
            raise ArgumentError, "Sum of input lengths does not equal the length of the input dataset!"
          end

          indices = Torch.randperm(lengths.sum).to_a
          _accumulate(lengths).zip(lengths).map { |offset, length| Subset.new(dataset, indices[(offset - length)...offset]) }
        end

        private

        def _accumulate(iterable)
          sum = 0
          iterable.map { |x| sum += x }
        end
      end
    end
  end
end
