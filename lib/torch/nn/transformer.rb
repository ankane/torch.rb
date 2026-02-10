require_relative 'transformer_encoder_layer'
require_relative 'transformer_encoder'
require_relative 'transformer_decoder_layer'
require_relative 'transformer_decoder'

module Torch
  module NN
    class Transformer < Module
      def initialize(
        d_model: 512,
        nhead: 8,
        num_encoder_layers: 6,
        num_decoder_layers: 6,
        dim_feedforward: 2048,
        dropout: 0.1,
        activation: :relu,
        custom_encoder: nil,
        custom_decoder: nil,
        layer_norm_eps: 1e-5,
        batch_first: false
      )
        super()

        @encoder =
          if custom_encoder
            custom_encoder
          else
            encoder_layer = TransformerEncoderLayer.new(
              d_model, nhead,
              dim_feedforward: dim_feedforward, dropout: dropout, activation: activation,
              layer_norm_eps: layer_norm_eps, batch_first: batch_first
            )
            encoder_norm = LayerNorm.new(d_model, eps: layer_norm_eps)
            TransformerEncoder.new(encoder_layer, num_encoder_layers, norm: encoder_norm)
          end

        @decoder =
          if custom_decoder
            custom_decoder
          else
            decoder_layer = TransformerDecoderLayer.new(
              d_model, nhead,
              dim_feedforward: dim_feedforward, dropout: dropout, activation: activation,
              layer_norm_eps: layer_norm_eps, batch_first: batch_first
            )
            decoder_norm = LayerNorm.new(d_model, eps: layer_norm_eps)
            TransformerDecoder.new(decoder_layer, num_decoder_layers, norm: decoder_norm)
          end

        reset_parameters

        @d_model = d_model
        @nhead = nhead
        @batch_first = batch_first
      end

      attr_reader :d_model, :nhead, :encoder, :decoder

      def batch_first?
        !!@batch_first
      end

      def reset_parameters
        parameters.each { |p| Init.xavier_uniform!(p) if p.dim > 1 }
      end

      def forward(
        src,
        tgt,
        src_mask: nil,
        tgt_mask: nil,
        memory_mask: nil,
        src_key_padding_mask: nil,
        tgt_key_padding_mask: nil,
        memory_key_padding_mask: nil
      )
        if (!batch_first? && src.size(1) != tgt.size(1)) ||
          (batch_first? && src.size(0) != tgt.size(0))

          raise ArgumentError, "The batch number of src and tgt must be equal"
        end

        if src.size(2) != d_model || tgt.size(2) != d_model
          raise ArgumentError, "The feature number of src and tgt must be equal to d_model"
        end

        memory = @encoder.(src, mask: src_mask, src_key_padding_mask: src_key_padding_mask)
        @decoder.(
          tgt, memory,
          tgt_mask: tgt_mask, memory_mask: memory_mask,
          tgt_key_padding_mask: tgt_key_padding_mask, memory_key_padding_mask: memory_key_padding_mask
        )
      end

      def generate_square_subsequent_mask(sz)
        mask = Torch.triu(Torch.ones([sz, sz])).eq(1).transpose(0, 1)
        mask.float.masked_fill!(mask.eq(0), -Float::INFINITY).masked_fill!(mask.eq(1), 0.0)
      end
    end
  end
end
