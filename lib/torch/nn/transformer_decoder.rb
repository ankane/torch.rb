module Torch
  module NN
    class TransformerDecoder < Module
      def initialize(decoder_layer, num_layers, norm: nil)
        super()

        @layers = _clones(decoder_layer, num_layers)
        @num_layers = num_layers
        @norm = norm
      end

      def forward(tgt, memory, tgt_mask: nil, memory_mask: nil, tgt_key_padding_mask: nil, memory_key_padding_mask: nil)
        output = tgt

        @layers.each do |mod|
          output = mod.call(output, memory, tgt_mask: tgt_mask, memory_mask: memory_mask, tgt_key_padding_mask: tgt_key_padding_mask, memory_key_padding_mask: memory_key_padding_mask)
        end

        output = @norm.call(output) if @norm

        output
      end
    end
  end
end
