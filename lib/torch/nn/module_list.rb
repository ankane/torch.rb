module Torch
  module NN
    class ModuleList < Module
      include Enumerable

      def initialize(mods = nil)
        super()

        concat(mods) if mods
      end

      def length
        @modules.length
      end
      alias_method :count, :length
      alias_method :size, :length

      def concat(mods)
        raise ArgumentError, "Modules should respond to #each" unless mods.respond_to?(:each)

        mods.each { |m| append m }

        self
      end

      def each(&block)
        if block_given?
          @modules.values.each(&block)
        else
          to_enum(:each)
        end
      end

      def map(&block)
        @modules.values.map(&block)
      end

      def append(mod)
        raise ArgumentError, "Provided element is not a module" unless mod.is_a?(Module)
        add_module(length.to_s, mod)
        self
      end

      def [](idx)
        if idx.is_a?(Range)
          self.class.new(@modules.values[idx])
        else
          @modules[idx.to_s]
        end
      end
    end
  end
end
