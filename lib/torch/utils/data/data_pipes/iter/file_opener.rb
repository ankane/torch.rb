module Torch
  module Utils
    module Data
      module DataPipes
        module Iter
          class FileOpener < IterDataPipe
            def initialize(datapipe, mode: "r", encoding: nil, length: -1)
              super()
              @datapipe = datapipe
              @mode = mode
              @encoding = encoding

              if !["b", "t", "rb", "rt", "r"].include?(@mode)
                raise ArgumentError, "Invalid mode #{mode}"
              end

              if mode.include?("b") && !encoding.nil?
                raise ArgumentError, "binary mode doesn't take an encoding argument"
              end

              @length = length
            end

            def each(&block)
              get_file_binaries_from_pathnames(@datapipe, @mode, encoding: @encoding, &block)
            end

            private

            def get_file_binaries_from_pathnames(pathnames, mode, encoding: nil)
              if !pathnames.is_a?(Enumerable)
                pathnames = [pathnames]
              end

              if ["b", "t"].include?(mode)
                mode = "r#{mode}"
              end

              pathnames.each do |pathname|
                if !pathname.is_a?(String)
                  raise TypeError, "Expected string type for pathname, but got #{pathname.class.name}"
                end
                yield pathname, StreamWrapper.new(File.open(pathname, mode, encoding: encoding))
              end
            end
          end
        end
      end
    end
  end
end
