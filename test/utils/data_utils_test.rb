require_relative "../test_helper"

class DataUtilsTest < Minitest::Test
  def test_works
    x = Torch.tensor([[1, 2], [3, 4], [5, 6]])
    y = Torch.tensor([5, 10, 15])
    dataset = Torch::Utils::Data::TensorDataset.new(x, y)
    assert_equal 3, dataset.size

    loader = Torch::Utils::Data::DataLoader.new(dataset, batch_size: 2)
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

  def test_tensor_dataset_bad
    x = Torch.tensor([[1, 2], [3, 4]])
    y = Torch.tensor([5])

    error = assert_raises Torch::Error do
      Torch::Utils::Data::TensorDataset.new(x, y)
    end
    assert_equal "Tensors must all have same dim 0 size", error.message
  end
end
