require_relative "test_helper"
require "socket"

class DistributedTest < Minitest::Test
  def setup
    super
    skip "Distributed backend not available" unless Torch::Distributed.available?
  end

  def test_all_reduce
    results = Torch::Distributed.fork_world(2) do |rank, port|
      store = Torch::Distributed::TCPStore.new("127.0.0.1", port, 2, rank.zero?)
      Torch::Distributed.init_process_group("gloo", store: store, rank: rank, world_size: 2)

      tensor = Torch.tensor([rank + 1.0])
      Torch::Distributed.all_reduce(tensor)
      Torch::Distributed.destroy_process_group
      tensor.to_a
    end

    assert_equal [[3.0], [3.0]], results
  end

  def test_barrier
    wait_times = Torch::Distributed.fork_world(2) do |rank, port|
      store = Torch::Distributed::TCPStore.new("127.0.0.1", port, 2, rank.zero?)
      Torch::Distributed.init_process_group("gloo", store: store, rank: rank, world_size: 2)

      sleep 0.3 if rank.zero?
      before = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      Torch::Distributed.barrier
      after = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      Torch::Distributed.destroy_process_group
      after - before
    end

    assert_operator wait_times.first, :<, 0.1
    assert_operator wait_times.last, :>=, 0.25
  end

  def test_broadcast
    tensors = Torch::Distributed.fork_world(2) do |rank, port|
      store = Torch::Distributed::TCPStore.new("127.0.0.1", port, 2, rank.zero?)
      Torch::Distributed.init_process_group("gloo", store: store, rank: rank, world_size: 2)

      tensor = Torch.tensor([rank + 1.0])
      Torch::Distributed.broadcast(tensor, src: 0)
      Torch::Distributed.destroy_process_group
      tensor.to_a
    end

    assert_equal [[1.0], [1.0]], tensors
  end

  def test_ddp_gradient_sync
    grads = Torch::Distributed.fork_world(2) do |rank, port|
      store = Torch::Distributed::TCPStore.new("127.0.0.1", port, 2, rank.zero?)
      Torch::Distributed.init_process_group("gloo", store: store, rank: rank, world_size: 2)

      model = Torch::NN::Linear.new(1, 1, bias: false)
      ddp = Torch::NN::Parallel::DistributedDataParallel.new(model)
      input = Torch.tensor([[rank + 1.0]])
      output = ddp.call(input)
      loss = output.sum
      loss.backward

      grad = model.parameters.first.grad.item
      Torch::Distributed.destroy_process_group
      grad
    end

    grads.each do |grad|
      assert_in_delta 1.5, grad, 1e-6
    end
  end

end
