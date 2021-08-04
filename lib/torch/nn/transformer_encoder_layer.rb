module Torch
  module NN
    class TransformerEncoderLayer < Module
      def initialize(
        d_model, n_head,
        dim_feedforward: 2048, dropout: 0.1, activation: :relu,
        layer_norm_eps: 1e-5, batch_first: false
      )

        super()

        @self_attn = MultiheadAttention.new(d_model, n_head, dropout: dropout, batch_first: batch_first)
        @linear1 = Linear.new(d_model, dim_feedforward)
        @dropout = Dropout.new(p: dropout)
        @linear2 = Linear.new(dim_feedforward, d_model)

        @norm1 = LayerNorm.new(d_model, eps: layer_norm_eps)
        @norm2 = LayerNorm.new(d_model, eps: layer_norm_eps)

        @dropout1 = Dropout.new(p: dropout)
        @dropout2 = Dropout.new(p: dropout)

        @activation = _activation_fn(activation)
      end

      def forward(src, src_mask: nil, src_key_padding_mask: nil)
        src2 = @self_attn.(src, src, src, attn_mask: src_mask, key_padding_mask: src_key_padding_mask).first
        src += @dropout1.(src2)
        src = @norm1.(src)
        src2 = @linear2.(@dropout.(@activation.(@linear1.(src))))
        src += @dropout2.(src2)
        @norm2.(src)
      end
    end
  end
end
