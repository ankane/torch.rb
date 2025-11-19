require_relative "test_helper"
require "torch/distributed"
require "socket"

class DistributedInitProcessGroupTest < Minitest::Test
  def setup
    skip "Distributed backend not available" unless Torch::Distributed.available?
    skip "CUDA not available for NCCL backend" unless cuda_available?
  end

  def test_defaults_nccl_device_id_from_local_rank_env
    calls = []
    with_stubbed_init_process_group(calls) do
      ENV["LOCAL_RANK"] = "2"
      Torch::Distributed.init_process_group("nccl", store: Object.new, rank: 5, world_size: 8)
    ensure
      ENV.delete("LOCAL_RANK")
    end

    assert_equal 1, calls.size
    assert_equal 2, calls.first[:device_id]
  end

  def test_falls_back_to_local_world_size_modulo
    calls = []
    with_stubbed_init_process_group(calls) do
      ENV["LOCAL_WORLD_SIZE"] = "2"
      Torch::Distributed.init_process_group("nccl", store: Object.new, rank: 3, world_size: 4)
    ensure
      ENV.delete("LOCAL_WORLD_SIZE")
    end

    assert_equal 1, calls.size
    assert_equal 1, calls.first[:device_id]
  end

  def test_uses_world_size_when_env_missing
    calls = []
    with_stubbed_init_process_group(calls) do
      Torch::Distributed.init_process_group("nccl", store: Object.new, rank: 1, world_size: 2)
    end

    assert_equal 1, calls.size
    assert_equal 1, calls.first[:device_id]
  end

  private

  def with_stubbed_init_process_group(calls)
    original = Torch::Distributed.method(:_init_process_group)
    Torch::Distributed.singleton_class.define_method(:_init_process_group) do |backend, store, rank, world_size, timeout_ms, device_id|
      calls << {backend: backend, rank: rank, world_size: world_size, timeout_ms: timeout_ms, device_id: device_id}
      :stub
    end
    yield
  ensure
    Torch::Distributed.singleton_class.define_method(:_init_process_group, original)
  end

  def cuda_available?
    Torch.const_defined?(:CUDA) && Torch::CUDA.respond_to?(:available?) && Torch::CUDA.available?
  end
end

class DistributedSpawnStartMethodTest < Minitest::Test
  def test_spawn_worker_env_runs_block
    reader, writer = IO.pipe
    writer.close_on_exec = false

    pid = fork do
      reader.close
      ENV[Torch::Distributed::SPAWN_ENV_KEY] = "1"
      ENV[Torch::Distributed::SPAWN_RANK_ENV_KEY] = "0"
      ENV[Torch::Distributed::SPAWN_WORLD_SIZE_ENV_KEY] = "1"
      ENV[Torch::Distributed::SPAWN_PORT_ENV_KEY] = "1234"
      ENV[Torch::Distributed::SPAWN_PIPE_ENV_KEY] = writer.fileno.to_s
      Torch::Distributed.fork_world(1, start_method: :spawn) { |rank, port| [rank, port] }
    end

    writer.close
    result = Marshal.load(reader)
    reader.close

    _pid, status = Process.wait2(pid)
    assert status.success?
    assert_equal [0, 1234], result
  end
end

class DistributedBackendTest < Minitest::Test
  BACKEND = nil

  def setup
    super
    skip "Distributed backend not available" unless Torch::Distributed.available?
    skip "No backend configured for test" unless backend
    skip_unless_backend_available!
  end

  def backend
    self.class::BACKEND
  end

  def tensor_options
    {}
  end

  def skip_unless_backend_available!
    skip "#{backend} backend not available" unless backend_available?
  end

  def backend_available?
    port = Torch::Distributed.free_port
    store = Torch::Distributed::TCPStore.new("127.0.0.1", port, 1, true, wait_for_workers: false)
    Torch::Distributed.init_process_group(backend, store: store, rank: 0, world_size: 1)
    true
  rescue StandardError => e
    return false if e.message =~ /not available/i || e.message =~ /unsupported backend/i
    raise
  ensure
    Torch::Distributed.destroy_process_group if Torch::Distributed.initialized?
  end

  def nccl_device_id(rank)
    rank
  end

  def fork_with_backend(world_size: 2, start_method: :fork)
    original_filter = ENV[Torch::Distributed::SPAWN_TEST_ENV_KEY]
    original_script = ENV[Torch::Distributed::SPAWN_SCRIPT_ENV_KEY]
    ENV[Torch::Distributed::SPAWN_TEST_ENV_KEY] = name if start_method == :spawn
    ENV[Torch::Distributed::SPAWN_SCRIPT_ENV_KEY] = File.expand_path(__FILE__) if start_method == :spawn
    Torch::Distributed.fork_world(world_size, start_method: start_method) do |rank, port|
      store = Torch::Distributed::TCPStore.new("127.0.0.1", port, world_size, rank.zero?)
      device_id = backend == "nccl" ? nccl_device_id(rank) : nil
      Torch::Distributed.init_process_group(backend, store: store, rank: rank, world_size: world_size, device_id: device_id)
      begin
        yield(rank)
      ensure
        Torch::Distributed.destroy_process_group
      end
    end
  ensure
    ENV[Torch::Distributed::SPAWN_TEST_ENV_KEY] = original_filter
    ENV[Torch::Distributed::SPAWN_SCRIPT_ENV_KEY] = original_script
  end

  def test_all_reduce
    results = fork_with_backend do |rank|
      tensor = Torch.tensor([rank + 1.0], **tensor_options)
      Torch::Distributed.all_reduce(tensor)
      tensor.to_a
    end

    assert_equal [[3.0], [3.0]], results
  end

  def test_barrier
    wait_times = fork_with_backend do |rank|
      sleep 0.3 if rank.zero?
      before = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      Torch::Distributed.barrier
      after = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      after - before
    end

    assert_operator wait_times.first, :<, 0.1
    assert_operator wait_times.last, :>=, 0.25
  end

  def test_broadcast
    tensors = fork_with_backend do |rank|
      tensor = Torch.tensor([rank + 1.0], **tensor_options)
      Torch::Distributed.broadcast(tensor, src: 0)
      tensor.to_a
    end

    assert_equal [[1.0], [1.0]], tensors
  end

  def test_ddp_gradient_sync
    grads = fork_with_backend do |rank|
      device = tensor_options[:device]
      model = Torch::NN::Linear.new(1, 1, bias: false)
      model = model.to(device) if device
      ddp = Torch::NN::Parallel::DistributedDataParallel.new(model)
      input = Torch.tensor([[rank + 1.0]], **tensor_options)
      output = ddp.call(input)
      loss = output.sum
      loss.backward

      grad = model.parameters.first.grad
      grad = grad.to("cpu") if device
      grad.item
    end

    grads.each do |grad|
      assert_in_delta 1.5, grad, 1e-6
    end
  end
end

class DistributedGlooTest < DistributedBackendTest
  BACKEND = "gloo"
end

class DistributedNcclTest < DistributedBackendTest
  BACKEND = "nccl"

  def setup
    skip "CUDA not available for NCCL backend" unless Torch.const_defined?(:CUDA) && Torch::CUDA.available?
    skip "Need at least 2 CUDA devices for NCCL tests" unless Torch::CUDA.device_count >= 2
    super
  end

  def tensor_options
    {device: "cuda"}
  end

  def fork_with_backend(world_size: 2, start_method: :spawn)
    super(world_size: world_size, start_method: start_method)
  end
end
