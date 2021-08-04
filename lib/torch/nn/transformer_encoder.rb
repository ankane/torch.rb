module Torch
  module NN
    class TransformerEncoder < Module
      def initialize(encoder_layer, num_layers, norm: nil)
        super()

        @layers = _clones(encoder_layer, num_layers)
        @num_layers = num_layers
        @norm = norm
      end

      def forward(src, mask: nil, src_key_padding_mask: nil)
        output = src

        @layers.each do |mod|
          output = mod.call(output, src_mask: mask, src_key_padding_mask: src_key_padding_mask)
        end

        output = @norm.call(output) if @norm

        output
      end
    end
  end
end
