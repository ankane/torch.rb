module Torch
  module NN
    class ModuleList < Module
      def initialize(mods = nil)
        super()

        return unless mods
        self.extend(mods)
      end

      def length
        @modules.length
      end

      alias :count :length

      def extend(mods)
        raise ArgumentError, "Modules should respond to #each" unless mods.respond_to?(:each)

        mods.each { |m| append m }

        self
      end

      def each(&block)
        @modules.values.each &block
      end

      def map(&block)
        @modules.values.map &block
      end

      def inject(inj, &block)
        @modules.values.inject(inj, &block)
      end

      def append(mod)
        raise ArgumentError, "Provided element is not a module" unless mod.is_a?(Module)
        add_module(length.to_s, mod)
        self
      end

      def [](*idx)
        idx.map do |id|
          if id.is_a?(Integer)
            @modules[id.to_s]
          elsif id.is_a?(Range)
            id.each do |i|
              @modules[i.to_s]
            end
          end
        end.flatten
      end
    end
  end
end
