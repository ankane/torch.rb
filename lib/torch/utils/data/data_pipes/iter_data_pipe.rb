module Torch
  module Utils
    module Data
      module DataPipes
        class IterDataPipe < IterableDataset
          def self.functional_datapipe(name)
            IterDataPipe.register_datapipe_as_function(name, self)
          end

          def self.functions
            @functions ||= {}
          end

          def self.register_datapipe_as_function(function_name, cls_to_register)
            if functions.include?(function_name)
              raise Error, "Unable to add DataPipe function name #{function_name} as it is already taken"
            end

            function = lambda do |source_dp, *args, **options, &block|
              cls_to_register.new(source_dp, *args, **options, &block)
            end
            functions[function_name] = function

            define_method function_name do |*args, **options, &block|
              IterDataPipe.functions[function_name].call(self, *args, **options, &block)
            end
          end

          def reset
            # no-op, but subclasses can override
          end

          def each(&block)
            reset
            @source_datapipe.each(&block)
          end
        end
      end
    end
  end
end
