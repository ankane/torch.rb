module Torch
  module NN
    class ParameterList < Module
      include Enumerable

      def initialize(parameters)
        super()
        @initialized = true
        unless parameters.nil?
          concat(parameters)
        end
      end

      def length
        @parameters.length
      end
      alias_method :count, :length
      alias_method :size, :length

      def concat(parameters)
        unless parameters.is_a?(Enumerable)
          raise TypeError, "ParameterList#concat should be called with an enumerable, but got #{parameters.class.name}"
        end
        offset = length
        parameters.each_with_index do |param, i|
          register_parameter((offset + i).to_s, param)
        end
        self
      end

      def each(&block)
        if block_given?
          @parameters.values.each(&block)
        else
          to_enum(:each)
        end
      end

      def [](idx)
        if idx.is_a?(Range)
          self.class.new(@parameters.values[idx])
        else
          @parameters[idx.to_s]
        end
      end
    end
  end
end
