module Torch
  module Utils
    module Data
      module DataPipes
        class FilterIterDataPipe < IterDataPipe
          functional_datapipe :filter

          def initialize(datapipe, &block)
            @datapipe = datapipe
            @filter_fn = block
          end

          def each
            @datapipe.each do |data|
              filtered = return_if_true(data)
              if non_empty?(filtered)
                yield filtered
              else
                Iter::StreamWrapper.close_streams(data)
              end
            end
          end

          def return_if_true(data)
            condition = @filter_fn.call(data)

            data if condition
          end

          def non_empty?(data)
            !data.nil?
          end
        end
      end
    end
  end
end
