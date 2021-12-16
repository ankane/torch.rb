module Torch
  module NN
    class TransformerDecoderLayer < Module
      def initialize(
        d_model, n_head,
        dim_feedforward: 2048, dropout: 0.1, activation: :relu,
        layer_norm_eps: 1e-5, batch_first: false
      )

        super()

        @self_attn = MultiheadAttention.new(d_model, n_head, dropout: dropout, batch_first: batch_first)
        @multihead_attn = MultiheadAttention.new(d_model, n_head, dropout: dropout, batch_first: batch_first)

        @linear1 = Linear.new(d_model, dim_feedforward)
        @dropout = Dropout.new(p: dropout)
        @linear2 = Linear.new(dim_feedforward, d_model)

        @norm1 = LayerNorm.new(d_model, eps: layer_norm_eps)
        @norm2 = LayerNorm.new(d_model, eps: layer_norm_eps)
        @norm3 = LayerNorm.new(d_model, eps: layer_norm_eps)

        @dropout1 = Dropout.new(p: dropout)
        @dropout2 = Dropout.new(p: dropout)
        @dropout3 = Dropout.new(p: dropout)

        @activation = _activation_fn(activation)
      end

      def forward(tgt, memory, tgt_mask: nil, memory_mask: nil, tgt_key_padding_mask: nil, memory_key_padding_mask: nil)
        tgt2 = @self_attn.(tgt, tgt, tgt, attn_mask: tgt_mask, key_padding_mask: tgt_key_padding_mask).first
        tgt += @dropout1.(tgt2)
        tgt = @norm1.(tgt)
        tgt2 = @multihead_attn.(tgt, memory, memory, attn_mask: memory_mask, key_padding_mask: memory_key_padding_mask).first
        tgt += @dropout2.(tgt2)
        tgt = @norm2.(tgt)
        tgt2 = @linear2.(@dropout.(@activation.(@linear1.(tgt))))
        tgt += @dropout3.(tgt2)
        @norm3.(tgt)
      end
    end
  end
end
