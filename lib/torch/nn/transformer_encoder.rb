module Torch
  module NN
    class TransformerEncoder < Module
      def initialize(encoder_layer, num_layers, norm: nil)
        super()

        state = encoder_layer.state_dict
        layers = num_layers.times.map do |i|
          encoder_layer.clone.tap { |l| l.load_state_dict(state) }
        end
        @layers = ModuleList.new(layers)

        @num_layers = num_layers
        @norm = norm
      end

      def forward(src, mask: nil, src_key_padding_mask: nil)
        out = @layers.inject(src) { |q, l| l.(q, src_mask: mask, src_key_padding_mask: src_key_padding_mask) }
        @norm ? @norm.(out) : out
      end
    end
  end
end
