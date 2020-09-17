require_relative "../test_helper"

class DataUtilsTest < Minitest::Test
  def test_data_loader
    x = Torch.tensor([[1, 2], [3, 4], [5, 6]])
    y = Torch.tensor([5, 10, 15])
    dataset = Torch::Utils::Data::TensorDataset.new(x, y)
    assert_equal 3, dataset.size
    assert_equal 3, dataset.length
    assert_equal 3, dataset.count

    loader = Torch::Utils::Data::DataLoader.new(dataset, batch_size: 2)
    assert_equal 2, loader.size
    assert_equal 2, loader.length
    assert_equal 2, loader.count

    x_out = []
    y_out = []
    loader.each do |xb, yb|
      x_out << xb.to_a
      y_out << yb.to_a
    end
    assert_equal [[[1, 2], [3, 4]], [[5, 6]]], x_out
    assert_equal [[5, 10], [15]], y_out

    assert loader.dataset

    # make sure other enum functions work
    loader.each_with_index do |(xb, yb), i|
    end
  end

  def test_data_loader_shuffle
    x = Torch.tensor([[1, 2], [3, 4], [5, 6]])
    y = Torch.tensor([5, 10, 15])
    dataset = Torch::Utils::Data::TensorDataset.new(x, y)
    loader = Torch::Utils::Data::DataLoader.new(dataset, batch_size: 2, shuffle: true)
    loader.each do |xb, yb|
    end
  end

  def test_tensor_dataset_bad
    x = Torch.tensor([[1, 2], [3, 4]])
    y = Torch.tensor([5])

    error = assert_raises Torch::Error do
      Torch::Utils::Data::TensorDataset.new(x, y)
    end
    assert_equal "Tensors must all have same dim 0 size", error.message
  end

  def test_random_split
    s1, s2 = Torch::Utils::Data.random_split((0..9).to_a, [3, 7])
    assert_equal 3, s1.length
    assert_equal 7, s2.length
    assert_equal 45, (s1.to_a + s2.to_a).sum
  end
end
