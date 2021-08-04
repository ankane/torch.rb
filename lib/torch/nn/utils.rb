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

      def _clones(mod, n)
        state = mod.state_dict
        layers = n.times.map do |i|
          mod.clone.tap { |l| l.load_state_dict(state) }
        end
        ModuleList.new(layers)
      end

      def _activation_fn(activation)
        case activation.to_sym
        when :relu then F.method(:relu)
        when :gelu then F.method(:gelu)
        else raise ArgumentError, "Activation should be relu/gelu, not `#{activation}`"
        end
      end
    end
  end
end
