require_relative "test_helper"

class DataParallelTest < Minitest::Test
  def test_scatter_cpu
    input = Torch.arange(12).reshape(4, 3)
    devices = ["cpu", "cpu"]

    scattered = Torch::NN._scatter(input, devices, 0)

    assert_equal 2, scattered.length
    assert_equal [2, 3], scattered[0].shape
    assert_equal [2, 3], scattered[1].shape
    assert_equal [[0, 1, 2], [3, 4, 5]], scattered[0].to_a
    assert_equal [[6, 7, 8], [9, 10, 11]], scattered[1].to_a
  end

  def test_scatter_dim1
    input = Torch.arange(12).reshape(3, 4)
    devices = ["cpu", "cpu"]

    scattered = Torch::NN._scatter(input, devices, 1)

    assert_equal 2, scattered.length
    assert_equal [3, 2], scattered[0].shape
    assert_equal [3, 2], scattered[1].shape
  end

  def test_gather_cpu
    t1 = Torch.tensor([[0, 1, 2], [3, 4, 5]])
    t2 = Torch.tensor([[6, 7, 8], [9, 10, 11]])

    gathered = Torch::NN._gather([t1, t2], "cpu", 0)

    assert_equal [4, 3], gathered.shape
    assert_equal Torch.arange(12).reshape(4, 3).to_a, gathered.to_a
  end

  def test_gather_dim1
    t1 = Torch.tensor([[0, 1], [4, 5], [8, 9]])
    t2 = Torch.tensor([[2, 3], [6, 7], [10, 11]])

    gathered = Torch::NN._gather([t1, t2], "cpu", 1)

    assert_equal [3, 4], gathered.shape
    assert_equal [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], gathered.to_a
  end

  def test_scatter_gather_roundtrip
    input = Torch.randn(8, 4)
    devices = ["cpu", "cpu"]

    scattered = Torch::NN._scatter(input, devices, 0)
    gathered = Torch::NN._gather(scattered, "cpu", 0)

    assert_equal input.shape, gathered.shape
    assert Torch.allclose(input, gathered)
  end

  def test_scatter_cuda
    skip "CUDA not available" unless Torch::CUDA.available?
    skip "Multiple GPUs required" unless Torch::CUDA.device_count >= 2

    input = Torch.arange(8).reshape(4, 2).to("cuda:0")
    devices = ["cuda:0", "cuda:1"]

    scattered = Torch::NN._scatter(input, devices, 0)

    assert_equal 2, scattered.length
    assert_equal "cuda:0", scattered[0].device.to_s
    assert_equal "cuda:1", scattered[1].device.to_s
  end

  def test_gather_cuda
    skip "CUDA not available" unless Torch::CUDA.available?
    skip "Multiple GPUs required" unless Torch::CUDA.device_count >= 2

    t1 = Torch.tensor([[0, 1], [2, 3]], device: "cuda:0")
    t2 = Torch.tensor([[4, 5], [6, 7]], device: "cuda:1")

    gathered = Torch::NN._gather([t1, t2], "cuda:0", 0)

    assert_equal [4, 2], gathered.shape
    assert_equal "cuda:0", gathered.device.to_s
  end

  def test_data_parallel_forward_cuda
    skip "CUDA not available" unless Torch::CUDA.available?
    skip "Multiple GPUs required" unless Torch::CUDA.device_count >= 2

    model = Torch::NN::Linear.new(10, 5).to("cuda:0")
    dp_model = Torch::NN::DataParallel.new(model, device_ids: [0, 1])

    input = Torch.randn(8, 10, device: "cuda:0")
    output = dp_model.call(input)

    assert_equal [8, 5], output.shape
    assert_equal "cuda:0", output.device.to_s
  end

  def test_data_parallel_sequential_cuda
    skip "CUDA not available" unless Torch::CUDA.available?
    skip "Multiple GPUs required" unless Torch::CUDA.device_count >= 2

    model = Torch::NN::Sequential.new(
      Torch::NN::Linear.new(10, 20),
      Torch::NN::Linear.new(20, 5)
    ).to("cuda:0")

    dp_model = Torch::NN::DataParallel.new(model, device_ids: [0, 1])

    input = Torch.arange(80, dtype: :float32).reshape(8, 10).to("cuda:0")
    output = dp_model.call(input)

    assert_equal [8, 5], output.shape
    assert_equal "cuda:0", output.device.to_s
  end

  def test_data_parallel_backward_single_gpu
    skip "CUDA not available" unless Torch::CUDA.available?

    model = Torch::NN::Linear.new(10, 5).to("cuda:0")
    dp_model = Torch::NN::DataParallel.new(model, device_ids: [0])

    input = Torch.ones(8, 10, dtype: :float32, device: "cuda:0")
    target = Torch.ones(8, 5, dtype: :float32, device: "cuda:0")

    output = dp_model.call(input)
    loss = (output - target).pow(2).mean
    loss.backward

    assert model.weight.grad, "Weight gradient should exist"
    assert model.bias.grad, "Bias gradient should exist"
    assert_equal [5, 10], model.weight.grad.shape
  end

  def test_data_parallel_backward_multi_gpu
    skip "CUDA not available" unless Torch::CUDA.available?
    skip "Multiple GPUs required" unless Torch::CUDA.device_count >= 2

    model = Torch::NN::Linear.new(10, 5).to("cuda:0")
    dp_model = Torch::NN::DataParallel.new(model, device_ids: [0, 1])

    input = Torch.ones(8, 10, dtype: :float32, device: "cuda:0")
    target = Torch.ones(8, 5, dtype: :float32, device: "cuda:0")

    output = dp_model.call(input)
    loss = (output - target).pow(2).mean
    loss.backward

    assert model.weight.grad, "Weight gradient should exist"
    assert model.bias.grad, "Bias gradient should exist"
    assert_equal [5, 10], model.weight.grad.shape
  end

  def test_data_parallel_parameters
    skip "CUDA not available" unless Torch::CUDA.available?

    model = Torch::NN::Linear.new(10, 5).to("cuda:0")
    dp_model = Torch::NN::DataParallel.new(model, device_ids: [0])

    assert_equal model.parameters.size, dp_model.parameters.size
  end

  def test_data_parallel_train_eval
    skip "CUDA not available" unless Torch::CUDA.available?

    model = Torch::NN::Sequential.new(
      Torch::NN::Linear.new(10, 5),
      Torch::NN::Dropout.new(p: 0.5)
    ).to("cuda:0")

    dp_model = Torch::NN::DataParallel.new(model, device_ids: [0])

    dp_model.train
    assert model.instance_variable_get(:@training)

    dp_model.eval
    refute model.instance_variable_get(:@training)
  end

  def test_data_parallel_state_dict
    skip "CUDA not available" unless Torch::CUDA.available?

    model = Torch::NN::Linear.new(10, 5).to("cuda:0")
    dp_model = Torch::NN::DataParallel.new(model, device_ids: [0])

    state_dict = dp_model.state_dict
    assert state_dict.key?("weight")
    assert state_dict.key?("bias")
  end

  def test_data_parallel_wrapped_module
    skip "CUDA not available" unless Torch::CUDA.available?

    model = Torch::NN::Linear.new(10, 5).to("cuda:0")
    dp_model = Torch::NN::DataParallel.new(model, device_ids: [0])

    # Both accessors should return the same module
    assert_equal model, dp_model.module
    assert_equal model, dp_model.wrapped_module
  end

  def test_data_parallel_backward_method_multi_gpu
    skip "CUDA not available" unless Torch::CUDA.available?
    skip "Multiple GPUs required" unless Torch::CUDA.device_count >= 2

    model = Torch::NN::Linear.new(10, 5).to("cuda:0")
    dp_model = Torch::NN::DataParallel.new(model, device_ids: [0, 1])

    input = Torch.ones(8, 10, dtype: :float32, device: "cuda:0")
    target = Torch.zeros(8, 5, dtype: :float32, device: "cuda:0")

    output = dp_model.call(input)
    loss = (output - target).pow(2).mean

    dp_model.backward(scale: 1.0)
    loss.backward

    assert model.weight.grad, "Weight gradient should exist after backward"
    refute model.weight.grad.sum.item.zero?, "Weight gradient should be non-zero"
  end

  def test_reduce_gradients_accumulates_correctly
    skip "CUDA not available" unless Torch::CUDA.available?
    skip "Multiple GPUs required" unless Torch::CUDA.device_count >= 2

    model = Torch::NN::Linear.new(4, 2).to("cuda:0")
    dp_model = Torch::NN::DataParallel.new(model, device_ids: [0, 1])

    input = Torch.ones(4, 4, dtype: :float32, device: "cuda:0")
    dp_model.call(input)

    replicas = dp_model.instance_variable_get(:@replicas)
    assert_equal 2, replicas.size, "Expected 2 replicas"

    replicas[0].weight.grad = Torch.ones(2, 4, device: "cuda:0")
    replicas[1].weight.grad = Torch.ones(2, 4, device: "cuda:1") * 2

    dp_model.reduce_gradients

    expected_sum = 3.0 * 2 * 4
    actual_sum = model.weight.grad.sum.item

    assert_in_delta expected_sum, actual_sum, 0.01, "Gradients should be summed from all replicas"
  end

  def test_data_parallel_with_loss_returning_model
    skip "CUDA not available" unless Torch::CUDA.available?
    skip "Multiple GPUs required" unless Torch::CUDA.device_count >= 2

    model_with_loss = Class.new(Torch::NN::Module) do
      def initialize
        super()
        @linear = Torch::NN::Linear.new(10, 5)
      end

      def forward(x, targets: nil)
        output = @linear.call(x)
        if targets
          loss = (output - targets).pow(2).mean
          [output, loss]
        else
          output
        end
      end
    end.new.to("cuda:0")

    dp_model = Torch::NN::DataParallel.new(model_with_loss, device_ids: [0, 1])

    input = Torch.ones(8, 10, dtype: :float32, device: "cuda:0")
    targets = Torch.zeros(8, 5, dtype: :float32, device: "cuda:0")

    output, loss = dp_model.call(input, targets: targets)

    assert_equal [8, 5], output.shape
    assert_equal [], loss.shape

    dp_model.backward(scale: 1.0)

    linear = model_with_loss.instance_variable_get(:@linear)
    assert linear.weight.grad, "Weight gradient should exist"
  end
end
