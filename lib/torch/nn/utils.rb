module Torch
  module NN
    module Utils
      def _single(value)
        _ntuple(1, value)
      end

      def _pair(value)
        _ntuple(2, value)
      end

      def _triple(value)
        _ntuple(3, value)
      end

      def _quadrupal(value)
        _ntuple(4, value)
      end

      def _ntuple(n, value)
        value.is_a?(Array) ? value : [value] * n
      end
    end
  end
end
