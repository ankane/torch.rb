module Torch
  module Utils
    module Data
      module DataPipes
        module Iter
          class FileLister < IterDataPipe
            def initialize(
              root = ".",
              masks = "",
              recursive: false,
              abspath: false,
              non_deterministic: false,
              length: -1
            )
              super()
              if root.is_a?(String)
                root = [root]
              end
              if !root.is_a?(IterDataPipe)
                root = IterableWrapper.new(root)
              end
              @datapipe = root
              @masks = masks
              @recursive = recursive
              @abspath = abspath
              @non_deterministic = non_deterministic
              @length = length
            end

            def each(&block)
              @datapipe.each do |path|
                get_file_pathnames_from_root(path, @masks, recursive: @recursive, abspath: @abspath, non_deterministic: @non_deterministic, &block)
              end
            end

            private

            def get_file_pathnames_from_root(
              root,
              masks,
              recursive: false,
              abspath: false,
              non_deterministic: false
            )
              if File.file?(root)
                raise NotImplementedYet
              else
                pattern = recursive ? "**/*" : "*"
                paths = Dir.glob(pattern, base: root)
                paths = paths.sort if non_deterministic
                paths.each do |f|
                  if abspath
                    raise NotImplementedYet
                  end
                  if match_masks(f, masks)
                    yield File.join(root, f)
                  end
                end
              end
            end

            def match_masks(name, masks)
              if masks.empty?
                return true
              end

              raise NotImplementedYet
            end
          end
        end
      end
    end
  end
end
