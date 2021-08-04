module Torch
  module NN
    class Functional
      class << self
        def in_projection_packed(q, k, v, w, b: nil)
          e = q.size(-1)

          if k.eql? v
            if q.eql? k
              # self-attention
              return linear(q, w, b).chunk(3, dim: -1)
            else
              # encoder-decoder attention
              w_q, w_kv = w.split_with_sizes([e, e * 2])
              if b.nil?
                b_q = b_kv = nil
              else
                b_q, b_kv = b.split_with_sizes([e, e * 2])
              end

              return [linear(q, w_q, b_q), *linear(k, w_kv, b_kv).chunk(2, dim: -1)]
            end
          else
            w_q, w_k, w_v = w.chunk(3)
            if b.nil?
              b_q = b_k = b_v = nil
            else
              b_q, b_k, b_v = b.chunk(3)
            end

            return [linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)]
          end
        end

        def in_projection(
          q, k, v,
          w_q, w_k, w_v,
          b_q: nil, b_k: nil, b_v: nil
        )

          e_q, e_k, e_v = q.size(-1), k.size(-1), v.size(-1)

          raise ArgumentError, "Expecting query weights shape of #{[e_q, e_q]}, but got #{w_q.shape}" unless w_q.shape == [e_q, e_q]
          raise ArgumentError, "Expecting key weights shape of #{[e_k, e_k]}, but got #{w_k.shape}" unless w_k.shape == [e_k, e_k]
          raise ArgumentError, "Expecting value weights shape of #{[e_v, e_v]}, but got #{w_v.shape}" unless w_v.shape == [e_v, e_v]

          raise ArgumentError, "Expecting query bias shape of #{[e_q]}, but got #{b_q.shape}" if b_q && b_q.shape != [e_q]
          raise ArgumentError, "Expecting key bias shape of #{[e_k]}, but got #{b_k.shape}" if b_k && b_k.shape != [e_k]
          raise ArgumentError, "Expecting value bias shape of #{[e_v]}, but got #{b_v.shape}" if b_v && b_v.shape != [e_v]

          [linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)]
        end

        def scaled_dot_product_attention(
          q, k, v,
          attn_mask: nil, dropout_p: 0.0
        )

          b, nt, e = q.shape

          q = q / Math.sqrt(e)

          attn = Torch.bmm(q, k.transpose(-2, -1))
          attn += attn_mask if attn_mask
          attn = softmax(attn, dim: -1)
          attn = dropout(attn, p: dropout_p) if dropout_p > 0

          output = Torch.bmm(attn, v)

          [output, attn]
        end

        def multi_head_attention_forward(
          query, key, value,
          embed_dim_to_check, num_heads,
          in_proj_weight, in_proj_bias,
          bias_k, bias_v,
          add_zero_attn,
          dropout_p,
          out_proj_weight, out_proj_bias,
          training: true,
          key_padding_mask: nil,
          need_weights: true,
          attn_mask: nil,
          use_separate_proj_weight: false,
          q_proj_weight: nil, k_proj_weight: nil, v_proj_weight: nil,
          static_k: nil, static_v: nil
        )

          tgt_len, bsz, embed_dim = query.shape
          src_len = key.shape.first

          raise ArgumentError, "Was expecting embedding dimension of #{embed_dim_to_check}, but got #{embed_dim}" unless embed_dim == embed_dim_to_check

          head_dim = if embed_dim.is_a?(Torch::Tensor)
            embed_dim.div(num_heads, rounding_mode: 'trunc')
          else
            head_dim = embed_dim.div num_heads
          end

          if use_separate_proj_weight
            raise ArgumentError, "Key's sequence and batch dims #{key.shape[0...2]} do not match value's #{value.shape[0...2]}" unless key.shape[0...2] == value.shape[0...2]
          else
            raise ArgumentError, "Key shape #{key.shape} does not match value shape #{value.shape}" unless key.shape == value.shape
          end

          # compute in-projection
          q, k, v =
            if use_separate_proj_weight
              raise ArgumentError, "use_separate_proj_weight is true but q_proj_weight is nil" unless q_proj_weight
              raise ArgumentError, "use_separate_proj_weight is true but k_proj_weight is nil" unless k_proj_weight
              raise ArgumentError, "use_separate_proj_weight is true but v_proj_weight is nil" unless v_proj_weight

              if in_proj_bias
                b_q, b_k, b_v = in_proj_bias.chunk(3)
              else
                b_q = b_k = b_v = nil
              end

              in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q: b_q, b_k: b_k, b_v: b_v)
            else
              in_projection_packed(query, key, value, in_proj_weight, b: in_proj_bias)
            end

          # prep attention mask
          if attn_mask
            if attn_mask.dtype == :uint8
              puts "[WARN] Byte tensor for attn_mask in Multihead Attention is deprecated. Use bool tensor instead."
              attn_mask = attn_mask.bool
            else
              raise ArgumentError, "Only float, byte, and bool types are supported for attn_mask, not #{attn_mask.dtype}" unless attn_mask.floating_point? || attn_mask.dtype == :bool
            end

            if attn_mask.dim == 2
              correct_2d_size = [tgt_len, src_len]
              raise ArgumentError, "The shape of the 2D attn_mask is #{attn_mask.shape}, but should be #{correct_2d_size}." unless attn_mask.shape == correct_2d_size

              attn_mask = attn_mask.unsqueeze(0)
            elsif attn_mask.dim == 3
              correct_3d_size = [bsz * num_heads, tgt_len, src_len]
              raise ArgumentError, "The shape of the 3D attn_mask is #{attn_mask.shape}, but should be #{correct_3d_size}." unless attn_mask.shape == correct_3d_size
            else
              raise ArgumentError, "attn_mask's dimension #{attn_mask.dim} is not supported"
            end
          end

          # prep key padding mask
          if key_padding_mask && key_padding_mask.dtype == :uint8
            puts "[WARN] Byte tensor for key_padding_mask in Multihead Attention is deprecated. Use bool tensor instead."
            key_padding_mask = key_padding_mask.bool
          end

          # add bias along batch dimension (currently second)
          if bias_k && bias_v
            raise ArgumentError, "bias cannot be added to static key." if static_k
            raise ArgumentError, "bias cannot be added to static value." if static_v

            k = Torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = Torch.cat([v, bias_v.repeat(1, bsz, 1)])

            attn_mask = pad(attn_mask, [0, 1]) if attn_mask
            key_padding_mask = pad(key_padding_mask, [0, 1]) if key_padding_mask
          else
            raise ArgumentError unless bias_k.nil?
            raise ArgumentError unless bias_v.nil?
          end

          # reshape q, k, v for multihead attention and make em batch first
          q = q.contiguous.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

          if static_k.nil?
            k = k.contiguous.view(-1, bsz * num_heads, head_dim).transpose(0, 1)
          else
            raise ArgumentError, "Expecting static_k.size(0) of #{bsz * num_heads}, but got #{static_k.size(0)}" unless static_k.size(0) == bsz * num_heads
            raise ArgumentError, "Expecting static_k.size(2) of #{head_dim}, but got #{static_k.size(2)}" unless static_k.size(2) == head_dim

            k = static_k
          end

          if static_v.nil?
            v = v.contiguous.view(-1, bsz * num_heads, head_dim).transpose(0, 1)
          else
            raise ArgumentError, "Expecting static_v.size(0) of #{bsz * num_heads}, but got #{static_v.size(0)}" unless static_v.size(0) == bsz * num_heads
            raise ArgumentError, "Expecting static_v.size(2) of #{head_dim}, but got #{static_v.size(2)}" unless static_v.size(2) == head_dim

            v = static_v
          end

          # add zero attention along batch dimension (now first)
          if add_zero_attn
            zero_attn_shape = [bsz * num_heads, 1, head_dim]
            k = Torch.cat([k, Torch.zeros(zero_attn_shape, dtype: k.dtype, device: k.device)], dim: 1)
            v = Torch.cat([v, Torch.zeros(zero_attn_shape, dtype: v.dtype, device: v.device)], dim: 1)

            attn_mask = pad(attn_mask, [0, 1]) if attn_mask
            key_padding_mask = pad(key_padding_mask, [0, 1]) if key_padding_mask
          end

          # update source sequence length after adjustments
          src_len = k.size(1)

          # merge key padding and attention masks
          if key_padding_mask
            raise ArgumentError, "Expecting key_padding_mask shape of #{[bsz, src_len]}, but got #{key_padding_mask.shape}" unless key_padding_mask.shape == [bsz, src_len]

            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)

            attn_mask = if attn_mask.nil?
              key_padding_mask
            elsif attn_mask.dtype == :bool
              attn_mask.logical_or(key_padding_mask)
            else
              attn_mask.masked_fill(key_padding_mask, -Float::INFINITY)
            end
          end

          # convert mask to float
          if attn_mask && attn_mask.dtype == :bool
            new_attn_mask = Torch.zeros_like(attn_mask, dtype: :float32)
            attn_mask = new_attn_mask.masked_fill(attn_mask, -Float::INFINITY)
          end

          dropout_p = 0.0 unless training

          # (deep breath) calculate attention and out projection
          attn_output, attn_output_weights = scaled_dot_product_attention(q, k, v, attn_mask: attn_mask, dropout_p: dropout_p)
          attn_output = attn_output.transpose(0, 1).contiguous.view(tgt_len, bsz, embed_dim)
          attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

          if need_weights
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            [attn_output, attn_output_weights.sum(dim: 1) / num_heads]
          else
            [attn_output, nil]
          end
        end
      end
    end
  end
end
