require_relative "../test_helper"

class TranformerTest < Minitest::Test
  def test_transformer_encoder
    Torch.manual_seed(42)
    src = Torch.randn(8, 2, 6)
    layer = Torch::NN::TransformerEncoderLayer.new(6, 2)
    encoder = Torch::NN::TransformerEncoder.new(layer, 4)

    expected_keys = ['layers.0.self_attn.in_proj_weight', 'layers.0.self_attn.in_proj_bias', 'layers.0.self_attn.out_proj.weight', 'layers.0.self_attn.out_proj.bias', 'layers.0.linear1.weight', 'layers.0.linear1.bias', 'layers.0.linear2.weight', 'layers.0.linear2.bias', 'layers.0.norm1.weight', 'layers.0.norm1.bias', 'layers.0.norm2.weight', 'layers.0.norm2.bias', 'layers.1.self_attn.in_proj_weight', 'layers.1.self_attn.in_proj_bias', 'layers.1.self_attn.out_proj.weight', 'layers.1.self_attn.out_proj.bias', 'layers.1.linear1.weight', 'layers.1.linear1.bias', 'layers.1.linear2.weight', 'layers.1.linear2.bias', 'layers.1.norm1.weight', 'layers.1.norm1.bias', 'layers.1.norm2.weight', 'layers.1.norm2.bias', 'layers.2.self_attn.in_proj_weight', 'layers.2.self_attn.in_proj_bias', 'layers.2.self_attn.out_proj.weight', 'layers.2.self_attn.out_proj.bias', 'layers.2.linear1.weight', 'layers.2.linear1.bias', 'layers.2.linear2.weight', 'layers.2.linear2.bias', 'layers.2.norm1.weight', 'layers.2.norm1.bias', 'layers.2.norm2.weight', 'layers.2.norm2.bias', 'layers.3.self_attn.in_proj_weight', 'layers.3.self_attn.in_proj_bias', 'layers.3.self_attn.out_proj.weight', 'layers.3.self_attn.out_proj.bias', 'layers.3.linear1.weight', 'layers.3.linear1.bias', 'layers.3.linear2.weight', 'layers.3.linear2.bias', 'layers.3.norm1.weight', 'layers.3.norm1.bias', 'layers.3.norm2.weight', 'layers.3.norm2.bias']
    assert_equal Set.new(encoder.state_dict.keys), Set.new(expected_keys)

    out = encoder.(src).detach

    expected_out = Torch.tensor([
      [[ 0.7493,  0.4482, -2.1426,  0.5586,  0.5540, -0.1676],
       [-1.7787,  1.3332, -0.3269, -0.2184,  0.9501,  0.0408]],

      [[ 0.0258, -0.3633,  0.4725, -0.5102,  1.8175, -1.4423],
       [-0.8428,  0.8163, -1.7820,  0.9993,  0.1579,  0.6513]],

      [[-0.8899,  0.4441, -0.8299,  0.1568,  1.9144, -0.7954],
       [ 0.9666, -1.8733,  1.0490,  0.3950, -0.5475,  0.0102]],

      [[-0.7694,  1.4112, -0.7571, -0.2797,  1.3567, -0.9616],
       [-0.8945,  1.2717,  1.4981, -0.8380, -0.2971, -0.7402]],

      [[ 1.3992, -1.0341, -1.3842, -0.0247,  0.0162,  1.0276],
       [-0.8861,  0.9142, -0.5524,  0.8005,  1.1647, -1.4410]],

      [[ 0.1054, -1.9251, -0.0421,  0.2794,  1.4807,  0.1016],
       [-0.5518, -0.8835, -0.7934,  0.6458,  1.9350, -0.3522]],

      [[ 1.3186, -1.4948, -1.1052,  0.1480,  0.3011,  0.8324],
       [-1.0710,  1.1253, -1.0413, -0.5237,  1.4925,  0.0183]],

      [[ 0.9012, -1.3407,  0.7998, -0.7706, -0.8129,  1.2232],
       [ 0.5637, -1.5301,  1.0149,  1.2128, -0.7807, -0.4805]]
    ])

    assert_equal out.shape, expected_out.shape
    # assert (expected_out - out).abs.lt(1e-6).all.item
  end

  def test_transformer_decoder
    Torch.manual_seed(42)
    memory = Torch.randn([8, 2, 6])
    tgt = Torch.randn(4, 2, 6)
    layer = Torch::NN::TransformerDecoderLayer.new(6, 2)
    decoder = Torch::NN::TransformerDecoder.new(layer, 4)

    expected_keys = ['layers.0.self_attn.in_proj_weight', 'layers.0.self_attn.in_proj_bias', 'layers.0.self_attn.out_proj.weight', 'layers.0.self_attn.out_proj.bias', 'layers.0.multihead_attn.in_proj_weight', 'layers.0.multihead_attn.in_proj_bias', 'layers.0.multihead_attn.out_proj.weight', 'layers.0.multihead_attn.out_proj.bias', 'layers.0.linear1.weight', 'layers.0.linear1.bias', 'layers.0.linear2.weight', 'layers.0.linear2.bias', 'layers.0.norm1.weight', 'layers.0.norm1.bias', 'layers.0.norm2.weight', 'layers.0.norm2.bias', 'layers.0.norm3.weight', 'layers.0.norm3.bias', 'layers.1.self_attn.in_proj_weight', 'layers.1.self_attn.in_proj_bias', 'layers.1.self_attn.out_proj.weight', 'layers.1.self_attn.out_proj.bias', 'layers.1.multihead_attn.in_proj_weight', 'layers.1.multihead_attn.in_proj_bias', 'layers.1.multihead_attn.out_proj.weight', 'layers.1.multihead_attn.out_proj.bias', 'layers.1.linear1.weight', 'layers.1.linear1.bias', 'layers.1.linear2.weight', 'layers.1.linear2.bias', 'layers.1.norm1.weight', 'layers.1.norm1.bias', 'layers.1.norm2.weight', 'layers.1.norm2.bias', 'layers.1.norm3.weight', 'layers.1.norm3.bias', 'layers.2.self_attn.in_proj_weight', 'layers.2.self_attn.in_proj_bias', 'layers.2.self_attn.out_proj.weight', 'layers.2.self_attn.out_proj.bias', 'layers.2.multihead_attn.in_proj_weight', 'layers.2.multihead_attn.in_proj_bias', 'layers.2.multihead_attn.out_proj.weight', 'layers.2.multihead_attn.out_proj.bias', 'layers.2.linear1.weight', 'layers.2.linear1.bias', 'layers.2.linear2.weight', 'layers.2.linear2.bias', 'layers.2.norm1.weight', 'layers.2.norm1.bias', 'layers.2.norm2.weight', 'layers.2.norm2.bias', 'layers.2.norm3.weight', 'layers.2.norm3.bias', 'layers.3.self_attn.in_proj_weight', 'layers.3.self_attn.in_proj_bias', 'layers.3.self_attn.out_proj.weight', 'layers.3.self_attn.out_proj.bias', 'layers.3.multihead_attn.in_proj_weight', 'layers.3.multihead_attn.in_proj_bias', 'layers.3.multihead_attn.out_proj.weight', 'layers.3.multihead_attn.out_proj.bias', 'layers.3.linear1.weight', 'layers.3.linear1.bias', 'layers.3.linear2.weight', 'layers.3.linear2.bias', 'layers.3.norm1.weight', 'layers.3.norm1.bias', 'layers.3.norm2.weight', 'layers.3.norm2.bias', 'layers.3.norm3.weight', 'layers.3.norm3.bias']
    assert_equal Set.new(decoder.state_dict.keys), Set.new(expected_keys)

    out = decoder.(tgt, memory).detach

    expected_out = Torch.tensor([
      [[ 0.9910, -1.6614,  0.4585,  1.1229, -0.8866, -0.0244],
       [ 0.2247, -0.9688,  0.4191,  1.8912, -0.9096, -0.6565]],

      [[-0.0579, -0.8439,  1.1724,  0.8325,  0.5904, -1.6936],
       [ 0.7203, -0.9428,  1.3076,  0.3839,  0.1755, -1.6445]],

      [[ 1.1308, -1.1648,  0.9485,  0.5929, -0.0547, -1.4527],
       [-0.2060, -1.2025,  0.2268,  1.5961,  0.7484, -1.1629]],

      [[-0.2963, -0.6104,  1.0706,  1.4588, -0.1225, -1.5001],
       [ 0.8797, -1.1604,  0.9647,  0.8675, -0.0712, -1.4803]]
    ])

    assert_equal out.shape, expected_out.shape
    # assert (expected_out - out).abs.lt(1e-6).all.item
  end

  def test_entire_transformer
    Torch.manual_seed(42)
    src = Torch.randn([8, 2, 6])
    tgt = Torch.randn(4, 2, 6)

    tf = Torch::NN::Transformer.new(d_model: 6, nhead: 2)
    out = tf.(src, tgt).detach

    expected_out = Torch.tensor([
      [[ 1.3946,  1.0311, -0.4112, -1.4705, -0.7782,  0.2342],
       [ 1.3813,  0.7335,  0.4295, -1.7469, -0.3987, -0.3987]],

      [[ 0.8528,  0.2527,  1.0666, -1.0627,  0.5239, -1.6332],
       [ 1.0099,  0.6658,  1.2135, -1.2414, -1.1116, -0.5361]],

      [[ 0.7495,  0.7391,  1.1455, -1.5647, -0.0059, -1.0636],
       [ 0.6769, -0.6463,  1.1300, -0.6820,  1.0389, -1.5175]],

      [[ 1.0712,  0.8934,  0.2774, -1.7420,  0.3894, -0.8894],
       [ 0.9592,  0.6803,  1.0008, -1.6594, -0.0541, -0.9268]]
    ])

    assert_equal out.shape, expected_out.shape
    # assert (expected_out - out).abs.lt(1e-6).all.item
  end

  def test_generate_square_subsequent_mask
    transformer = Torch::NN::Transformer.new
    assert_equal [[0, -Float::INFINITY], [0, 0]], transformer.generate_square_subsequent_mask(2).to_a
  end
end
