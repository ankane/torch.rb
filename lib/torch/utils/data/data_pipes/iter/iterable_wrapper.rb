module Torch
  module Utils
    module Data
      module DataPipes
        module Iter
          class IterableWrapper < IterDataPipe
            def initialize(iterable, deepcopy: true)
              @iterable = iterable
              @deepcopy = deepcopy
            end

            def each
              source_data = @iterable
              if @deepcopy
                source_data = Marshal.load(Marshal.dump(@iterable))
              end
              source_data.each do |data|
                yield data
              end
            end

            def length
              @iterable.length
            end
          end
        end
      end
    end
  end
end
