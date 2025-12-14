require_relative "test_helper"

class ParallelApplyTest < Minitest::Test
  def test_parallel_apply_cpu
    model1 = Torch::NN::Linear.new(10, 5)
    model2 = Torch::NN::Linear.new(10, 5)

    # Copy weights so outputs are comparable
    model2.load_state_dict(model1.state_dict)

    inputs = [Torch.randn(2, 10), Torch.randn(2, 10)]

    outputs = Torch::NN::Parallel.parallel_apply([model1, model2], inputs)

    assert_equal 2, outputs.size
    assert_equal [2, 5], outputs[0].shape
    assert_equal [2, 5], outputs[1].shape
  end

  def test_parallel_apply_cuda
    skip "CUDA not available" unless Torch::CUDA.available?
    skip "Multiple GPUs required" unless Torch::CUDA.device_count >= 2

    model = Torch::NN::Linear.new(10, 5).to("cuda:0")
    replicas = Torch::NN::Parallel.replicate(model, ["cuda:0", "cuda:1"])

    # Use arange instead of randn to avoid kernel compatibility issues with older GPUs
    input0 = Torch.arange(20, dtype: :float32).reshape(2, 10).to("cuda:0")
    input1 = Torch.arange(20, dtype: :float32).reshape(2, 10).to("cuda:1")

    outputs = Torch::NN::Parallel.parallel_apply(replicas, [input0, input1])

    assert_equal 2, outputs.size
    assert_equal "cuda:0", outputs[0].device.to_s
    assert_equal "cuda:1", outputs[1].device.to_s
  end
end
