require_relative "../test_helper"

class FunctionalAttentionTest < Minitest::Test
  def test_self_attention_no_mask
    t = Torch.ones([4, 2, 6])
    Torch.manual_seed(42)
    attn = Torch::NN::MultiheadAttention.new(6, 2)
    out, weights = attn.(t, t, t)

    expected_out = Torch.tensor([
      [[-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357],
       [-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357]],

      [[-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357],
       [-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357]],

      [[-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357],
       [-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357]],

      [[-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357],
       [-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357]]
    ])

    expected_weights = Torch.tensor([
      [[0.2500, 0.2500, 0.2500, 0.2500],
       [0.2500, 0.2500, 0.2500, 0.2500],
       [0.2500, 0.2500, 0.2500, 0.2500],
       [0.2500, 0.2500, 0.2500, 0.2500]],

      [[0.2500, 0.2500, 0.2500, 0.2500],
       [0.2500, 0.2500, 0.2500, 0.2500],
       [0.2500, 0.2500, 0.2500, 0.2500],
       [0.2500, 0.2500, 0.2500, 0.2500]]
    ])

    assert_equal out.shape, expected_out.shape
    assert_equal weights.shape, expected_weights.shape

    [[out.detach, expected_out], [weights.detach, expected_weights]].each do |(a, b)|
      assert (a - b).abs.lt(1e-6).all
    end
  end

  def test_self_attention_with_masks
    t = Torch.ones([4, 2, 6])
    Torch.manual_seed(42)
    attn = Torch::NN::MultiheadAttention.new(6, 2)

    attn_mask = Torch.triu(Torch.ones([4, 4]), diagonal: 1).eq(1)
    key_padding_mask = Torch.triu(Torch.zeros(2, 4))
    key_padding_mask[0, -1] = 1

    out, weights = attn.(t, t, t, attn_mask: attn_mask, key_padding_mask: key_padding_mask)

    expected_out = Torch.tensor([
      [[-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357],
       [-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357]],

      [[-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357],
       [-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357]],

      [[-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357],
       [-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357]],

      [[-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357],
       [-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357]]
    ])

    expected_weights = Torch.tensor([
      [[1.0000, 0.0000, 0.0000, 0.0000],
       [0.5000, 0.5000, 0.0000, 0.0000],
       [0.3333, 0.3333, 0.3333, 0.0000],
       [0.3333, 0.3333, 0.3333, 0.0000]],

      [[1.0000, 0.0000, 0.0000, 0.0000],
       [0.5000, 0.5000, 0.0000, 0.0000],
       [0.3333, 0.3333, 0.3333, 0.0000],
       [0.2500, 0.2500, 0.2500, 0.2500]]
    ])

    assert_equal out.shape, expected_out.shape
    assert_equal weights.shape, expected_weights.shape

    [[out.detach, expected_out], [weights.detach, expected_weights]].each do |(a, b)|
      assert (a - b).abs.lt(1e-6).all
    end
  end

  def test_encoder_decoder_attention
    q = Torch.ones([4, 2, 6])
    k = v = Torch.ones([8, 2, 6])
    Torch.manual_seed(42)
    attn = Torch::NN::MultiheadAttention.new(6, 2)

    attn_mask = Torch.triu(Torch.ones([4, 8]), diagonal: 1).eq(1)
    key_padding_mask = Torch.triu(Torch.zeros(2, 8))
    key_padding_mask[0, -1] = 1

    out, weights = attn.(q, k, v, attn_mask: attn_mask, key_padding_mask: key_padding_mask)

    expected_out = Torch.tensor([
      [[-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357],
       [-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357]],

      [[-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357],
       [-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357]],

      [[-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357],
       [-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357]],

      [[-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357],
       [-1.2826,  0.4973, -0.3479,  0.3659,  0.6462,  0.1357]]
    ])

    expected_weights = Torch.tensor([
      [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
       [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
       [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
       [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000]],

      [[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
       [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
       [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
       [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000]]
    ])

    assert_equal out.shape, expected_out.shape
    assert_equal weights.shape, expected_weights.shape

    [[out.detach, expected_out], [weights.detach, expected_weights]].each do |(a, b)|
      assert (a - b).abs.lt(1e-6).all
    end
  end
end
