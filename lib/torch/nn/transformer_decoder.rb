module Torch
  module NN
    class TransformerDecoder < Module
      def initialize(decoder_layer, num_layers, norm: nil)
        super()

        state = decoder_layer.state_dict
        layers = num_layers.times.map do |i|
          decoder_layer.clone.tap { |l| l.load_state_dict(state) }
        end
        @layers = ModuleList.new(layers)

        @num_layers = num_layers
        @norm = norm
      end

      def forward(tgt, memory, tgt_mask: nil, memory_mask: nil, tgt_key_padding_mask: nil, memory_key_padding_mask: nil)
        out = @layers.inject(tgt) { |kv, l| l.(kv, memory, tgt_mask: tgt_mask, memory_mask: memory_mask, tgt_key_padding_mask: tgt_key_padding_mask, memory_key_padding_mask: memory_key_padding_mask) }
        @norm ? @norm.(out) : out
      end
    end
  end
end
