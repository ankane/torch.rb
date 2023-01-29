require_relative "test_helper"

class GeneratorTest < Minitest::Test
  def test_initial_seed
    g = Torch::Generator.new
    assert_kind_of Integer, g.initial_seed
  end

  def test_seed
    g = Torch::Generator.new
    assert_kind_of Integer, g.seed
  end

  def test_multinomial
    g = Torch::Generator.new.manual_seed(2147483647)
    t = Torch.tensor([0.2, 0.6, 0.2])
    ix = Torch.multinomial(t, num_samples: 1, replacement: true, generator: g).item
    assert_equal 1, ix
  end
end
