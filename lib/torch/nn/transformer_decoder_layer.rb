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

        @activation = activation_fn(activation)
      end

      def forward(tgt, memory, tgt_mask: nil, memory_mask: nil, tgt_key_padding_mask: nil, memory_key_padding_mask: nil)
        tmp = @self_attn.(tgt, tgt, tgt, attn_mask: tgt_mask, key_padding_mask: tgt_key_padding_mask).first
        out = tgt + @dropout1.(tmp)
        out = @norm1.(out)

        tmp = @multihead_attn.(tgt, memory, memory, attn_mask: memory_mask, key_padding_mask: memory_key_padding_mask).first
        out += @dropout2.(tmp)
        out = @norm2.(out)
        
        tmp = @activation.(@linear1.(out))
        tmp = @linear2.(@dropout.(tmp))
        out += @dropout2.(tmp)

        @norm3.(out)
      end

      private
      def activation_fn(activation)
        case activation.to_sym
        when :relu then F.method(:relu)
        when :gelu then F.method(:gelu)
        else raise ArgumentError, "Activation should be relu/gelu, not `#{activation}`"
        end
      end
    end
  end
end
