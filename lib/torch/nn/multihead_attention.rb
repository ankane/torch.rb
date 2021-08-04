module Torch
  module NN
    class MultiheadAttention < Module
      def initialize(
        embed_dim, num_heads,
        dropout: 0.0, bias: true, add_bias_kv: false, add_zero_attn: false,
        kdim: nil, vdim: nil, batch_first: false, device: nil, dtype: nil
      )

        super()

        @embed_dim = embed_dim
        @kdim = kdim || @embed_dim
        @vdim = vdim || @embed_dim

        @qkv_same_embed_dim = @kdim == @embed_dim && @vdim == @embed_dim

        @num_heads = num_heads
        @dropout = dropout
        @batch_first = batch_first

        @head_dim = @embed_dim.div @num_heads

        raise ArgumentError, "embed_dim must be divisible by num_heads" unless @head_dim * @num_heads == @embed_dim

        if @qkv_same_embed_dim
          @in_proj_weight = Parameter.new(Torch.empty([3 * @embed_dim, @embed_dim]))
          %w(q k v).each { |x| register_parameter("#{x}_proj_weight", nil) }
        else
          @q_proj_weight = Parameter.new(Torch.empty([@embed_dim, @embed_dim]))
          @k_proj_weight = Parameter.new(Torch.empty([@embed_dim, @kdim]))
          @v_proj_weight = Parameter.new(Torch.empty([@embed_dim, @vdim]))

          register_parameter('in_proj_weight', nil)
        end

        if bias
          @in_proj_bias = Parameter.new(Torch.empty(3 * @embed_dim))
        else
          register_parameter('in_proj_bias', nil)
        end

        @out_proj = Linear.new(@embed_dim, @embed_dim, bias: bias)

        if add_bias_kv
          @bias_k = Parameter.new(Torch.empty([1, 1, @embed_dim]))
          @bias_v = Parameter.new(Torch.empty([1, 1, @embed_dim]))
        else
          @bias_k = @bias_v = nil
        end

        @add_zero_attn = add_zero_attn

        reset_parameters
      end

      def batch_first?
        !!@batch_first
      end

      def reset_parameters
        if @qkv_same_embed_dim
          Init.xavier_uniform!(@in_proj_weight)
        else
          Init.xavier_uniform!(@q_proj_weight)
          Init.xavier_uniform!(@k_proj_weight)
          Init.xavier_uniform!(@v_proj_weight)
        end

        if @in_proj_bias
          Init.constant!(@in_proj_bias, 0.0)
          Init.constant!(@out_proj.bias, 0.0)
        end

        Init.xavier_uniform!(@bias_k) if @bias_k
        Init.xavier_uniform!(@bias_v) if @bias_v
      end

      def forward(
        query, key, value,
        key_padding_mask: nil, need_weights: true, attn_mask: nil
      )

        if batch_first?
          query, key, value = [query, key, value].map { |t| t.transpose(1, 0) }
        end

        attn_output, attn_output_weights =
          if @qkv_same_embed_dim
            F.multi_head_attention_forward(
              query, key, value,
              @embed_dim, @num_heads,
              @in_proj_weight, @in_proj_bias,
              @bias_k, @bias_v, @add_zero_attn,
              @dropout, @out_proj.weight, @out_proj.bias,
              training: @training,
              key_padding_mask: key_padding_mask,
              need_weights: need_weights,
              attn_mask: attn_mask
            )
          else
            F.multi_head_attention_forward(
              query, key, value,
              @embed_dim, @num_heads,
              @in_proj_weight, @in_proj_bias,
              @bias_k, @bias_v, @add_zero_attn,
              @dropout, @out_proj.weight, @out_proj.bias,
              training: @training,
              key_padding_mask: key_padding_mask,
              need_weights: need_weights,
              attn_mask: attn_mask,
              use_separate_proj_weight: true,
              q_proj_weight: @q_proj_weight, k_proj_weight: @k_proj_weight, v_proj_weight: @v_proj_weight
            )
          end

        attn_output = attn_output.transpose(1, 0) if batch_first?

        [attn_output, attn_output_weights]
      end
    end
  end
end
