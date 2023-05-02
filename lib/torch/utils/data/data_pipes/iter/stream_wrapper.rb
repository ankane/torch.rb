module Torch
  module Utils
    module Data
      module DataPipes
        module Iter
          class StreamWrapper
            def initialize(file_obj)
              @file_obj = file_obj
            end

            def gets(...)
              @file_obj.gets(...)
            end

            def close
              @file_obj.close
            end

            # TODO improve
            def self.close_streams(cls)
              if cls.is_a?(StreamWrapper)
                cls.close
              end
            end
          end
        end
      end
    end
  end
end
