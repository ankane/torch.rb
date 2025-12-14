require_relative "test_helper"

class ReplicateTest < Minitest::Test
  def test_replicate_linear_cpu
    model = Torch::NN::Linear.new(10, 5)
    devices = ["cpu", "cpu"]

    replicas = Torch::NN::Parallel.replicate(model, devices)

    assert_equal 2, replicas.size
    replicas.each do |replica|
      assert_instance_of Torch::NN::Linear, replica
      assert_equal [5, 10], replica.weight.shape
      assert_equal [5], replica.bias.shape
    end

    # Check weights are equal
    assert Torch.allclose(replicas[0].weight, replicas[1].weight)
    assert Torch.allclose(replicas[0].bias, replicas[1].bias)
  end

  def test_replicate_sequential_cpu
    model = Torch::NN::Sequential.new(
      Torch::NN::Linear.new(10, 5),
      Torch::NN::ReLU.new,
      Torch::NN::Linear.new(5, 2)
    )
    devices = ["cpu", "cpu"]

    replicas = Torch::NN::Parallel.replicate(model, devices)

    assert_equal 2, replicas.size
    replicas.each do |replica|
      assert_instance_of Torch::NN::Sequential, replica
      assert_equal 3, replica.children.size
    end

    # Check that replicas produce same output
    input = Torch.randn(4, 10)
    out0 = replicas[0].call(input)
    out1 = replicas[1].call(input)
    assert Torch.allclose(out0, out1)
  end

  def test_replicate_cuda
    skip "CUDA not available" unless Torch::CUDA.available?
    skip "Multiple GPUs required" unless Torch::CUDA.device_count >= 2

    model = Torch::NN::Linear.new(10, 5).to("cuda:0")
    devices = ["cuda:0", "cuda:1"]

    replicas = Torch::NN::Parallel.replicate(model, devices)

    assert_equal 2, replicas.size
    assert_equal "cuda:0", replicas[0].weight.device.to_s
    assert_equal "cuda:1", replicas[1].weight.device.to_s

    # Check weights are equal (after moving to same device)
    assert Torch.allclose(replicas[0].weight, replicas[1].weight.to("cuda:0"))
  end
end
