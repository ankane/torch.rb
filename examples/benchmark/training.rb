# Benchmark training throughput for common architectures/datasets.
# Usage examples:
#   ruby examples/benchmark/training.rb --arch mnist_cnn --batch-size 128 --gpus 1
#   ruby examples/benchmark/training.rb --arch mnist_cnn --batch-size 128 --gpus 2 --steps 50

require "bundler/setup"
require "optparse"
require "torch"
require "torchvision"

DEFAULT_BACKEND = if Torch.const_defined?(:CUDA) && Torch::CUDA.respond_to?(:available?) && Torch::CUDA.available?
  "nccl"
else
  Torch::Distributed.get_default_backend_for_device(Torch::Accelerator.current_accelerator) || "gloo"
end
SPAWN_BACKEND_ENV = "TORCH_RB_BENCH_BACKEND".freeze
SPAWN_GROUP_ENV = "TORCH_RB_BENCH_GROUP_SIZE".freeze
SPAWN_BATCH_ENV = "TORCH_RB_BENCH_BATCH_SIZE".freeze

def parse_list(value)
  value.split(",").map(&:strip).reject(&:empty?)
end

def backend_supported?(backend)
  return true unless backend == "nccl"

  Torch.const_defined?(:CUDA) && Torch::CUDA.respond_to?(:available?) && Torch::CUDA.available?
end

def usable_cuda_device_count
  return 0 unless Torch.const_defined?(:CUDA) && Torch::CUDA.respond_to?(:available?) && Torch::CUDA.available?

  Torch::CUDA.respond_to?(:device_count) ? Torch::CUDA.device_count : 0
rescue
  0
end

def spawn_worker_process?
  ENV[Torch::Distributed::SPAWN_ENV_KEY] == "1"
end

def apply_spawn_overrides!(options)
  return unless ENV[Torch::Distributed::SPAWN_ENV_KEY] == "1"

  if ENV[SPAWN_BACKEND_ENV]
    options[:backends] = [ENV[SPAWN_BACKEND_ENV]]
  end

  if ENV[SPAWN_GROUP_ENV]
    group_size = ENV[SPAWN_GROUP_ENV].to_i
    if group_size.positive?
      options[:group_sizes] = [group_size]
      options[:gpus] = group_size
    end
  end

  if ENV[SPAWN_BATCH_ENV]
    batch_size = ENV[SPAWN_BATCH_ENV].to_i
    options[:batch_sizes] = [batch_size] if batch_size.positive?
  end
end

def with_spawn_env(backend:, group_size:, batch_size:)
  previous = {
    SPAWN_BACKEND_ENV => ENV[SPAWN_BACKEND_ENV],
    SPAWN_GROUP_ENV => ENV[SPAWN_GROUP_ENV],
    SPAWN_BATCH_ENV => ENV[SPAWN_BATCH_ENV]
  }

  ENV[SPAWN_BACKEND_ENV] = backend
  ENV[SPAWN_GROUP_ENV] = group_size.to_s
  ENV[SPAWN_BATCH_ENV] = batch_size.to_s

  yield
ensure
  ENV[SPAWN_BACKEND_ENV] = previous[SPAWN_BACKEND_ENV]
  ENV[SPAWN_GROUP_ENV] = previous[SPAWN_GROUP_ENV]
  ENV[SPAWN_BATCH_ENV] = previous[SPAWN_BATCH_ENV]
end

class MnistCnn < Torch::NN::Module
  def initialize
    super()
    @conv1 = Torch::NN::Conv2d.new(1, 32, 3, stride: 1)
    @conv2 = Torch::NN::Conv2d.new(32, 64, 3, stride: 1)
    @dropout1 = Torch::NN::Dropout2d.new(p: 0.25)
    @dropout2 = Torch::NN::Dropout2d.new(p: 0.5)
    @fc1 = Torch::NN::Linear.new(9216, 128)
    @fc2 = Torch::NN::Linear.new(128, 10)
  end

  def forward(x)
    x = Torch::NN::F.relu(@conv1.call(x))
    x = Torch::NN::F.relu(@conv2.call(x))
    x = Torch::NN::F.max_pool2d(x, 2)
    x = @dropout1.call(x)
    x = Torch.flatten(x, start_dim: 1)
    x = Torch::NN::F.relu(@fc1.call(x))
    x = @dropout2.call(x)
    Torch::NN::F.log_softmax(@fc2.call(x), 1)
  end
end

ARCH_CONFIGS = {
  "mnist_cnn" => {
    model: -> { MnistCnn.new },
    dataset: :mnist
  }
}.freeze

def parse_options
  defaults = {
    arch: "mnist_cnn",
    batch_sizes: [128],
    steps: 100,
    warmup: 10,
    backends: [DEFAULT_BACKEND],
    gpus: Torch::CUDA.available? ? [Torch::CUDA.device_count, 1].max : 1,
    group_sizes: nil,
    data_dir: File.join(__dir__, "data"),
    lr: 0.01
  }

  OptionParser.new do |opts|
    opts.banner = "Usage: ruby examples/benchmark/training.rb [options]"
    opts.on("--arch NAME", "Architecture to benchmark (#{ARCH_CONFIGS.keys.join(', ')}, default: #{defaults[:arch]})") { |v| defaults[:arch] = v }
    opts.on("--batch-size N", Integer, "Batch size per process (default: #{defaults[:batch_sizes].first})") { |v| defaults[:batch_sizes] = [v] }
    opts.on("--batch-sizes LIST", String, "Comma-separated batch sizes per process") { |v| defaults[:batch_sizes] = parse_list(v).map(&:to_i) }
    opts.on("--steps N", Integer, "Number of timed training steps (default: #{defaults[:steps]})") { |v| defaults[:steps] = v }
    opts.on("--warmup N", Integer, "Number of warmup steps not included in timing (default: #{defaults[:warmup]})") { |v| defaults[:warmup] = v }
    opts.on("--backend NAME", String, "Process group backend (default: #{defaults[:backends].first})") { |v| defaults[:backends] = [v] }
    opts.on("--backends LIST", String, "Comma-separated list of backends to benchmark (gloo,nccl)") { |v| defaults[:backends] = parse_list(v) }
    opts.on("--gpus N", Integer, "Number of GPUs/processes to use (1 for non-distributed)") { |v| defaults[:gpus] = v }
    opts.on("--group-sizes LIST", String, "Process group sizes to benchmark (default: 1..gpus)") { |v| defaults[:group_sizes] = parse_list(v).map(&:to_i) }
    opts.on("--data-dir PATH", String, "Directory for cached datasets (default: #{defaults[:data_dir]})") { |v| defaults[:data_dir] = v }
    opts.on("--lr FLOAT", Float, "Learning rate (default: #{defaults[:lr]})") { |v| defaults[:lr] = v }
  end.parse!(ARGV)

  defaults[:group_sizes] ||= (1..defaults[:gpus]).to_a
  defaults
end

def dataset_for(name, data_dir, distributed:, rank:, world_size:)
  case name
  when :mnist
    transforms = TorchVision::Transforms::Compose.new([
      TorchVision::Transforms::ToTensor.new,
      TorchVision::Transforms::Normalize.new([0.1307], [0.3081])
    ])

    if distributed
      if rank.zero?
        train = TorchVision::Datasets::MNIST.new(data_dir, train: true, download: true, transform: transforms)
        Torch::Distributed.barrier
      else
        Torch::Distributed.barrier
        train = TorchVision::Datasets::MNIST.new(data_dir, train: true, download: false, transform: transforms)
      end
      indices = rank.step(train.size - 1, world_size).to_a
      Torch::Utils::Data::Subset.new(train, indices)
    else
      TorchVision::Datasets::MNIST.new(data_dir, train: true, download: true, transform: transforms)
    end
  else
    raise ArgumentError, "Unknown dataset: #{name}"
  end
end

def sync_cuda_if_needed(device)
  return unless device && device.type == "cuda"
  return unless Torch.const_defined?(:CUDA) && Torch::CUDA.respond_to?(:synchronize)

  Torch::CUDA.synchronize
end

def benchmark_worker(rank, world_size, port, options)
  arch = options.fetch(:arch)
  config = ARCH_CONFIGS[arch]
  raise ArgumentError, "Unsupported architecture #{arch.inspect}" unless config

  distributed = world_size > 1
  accelerator = Torch::Accelerator.current_accelerator
  selected_backend = options[:backend] || Torch::Distributed.get_default_backend_for_device(accelerator) || DEFAULT_BACKEND
  if distributed
    store = Torch::Distributed::TCPStore.new("127.0.0.1", port, world_size, rank.zero?)
    Torch::Distributed.init_process_group(selected_backend, store: store, rank: rank, world_size: world_size)
  end

  cuda_devices = usable_cuda_device_count
  device = if cuda_devices.positive? && options[:gpus] > 0
    Torch.device("cuda:#{rank % cuda_devices}")
  else
    Torch.device("cpu")
  end

  model = config[:model].call.to(device)
  if distributed
    ddp_devices = device.type == "cuda" ? [device.index] : nil
    model = Torch::NN::Parallel::DistributedDataParallel.new(model, device_ids: ddp_devices)
  end
  optimizer = Torch::Optim::SGD.new(model.parameters, lr: options[:lr])

  loader = Torch::Utils::Data::DataLoader.new(
    dataset_for(config[:dataset], options[:data_dir], distributed: distributed, rank: rank, world_size: world_size),
    batch_size: options[:batch_size],
    shuffle: true
  )

  warmup_steps = options[:warmup]
  timed_steps = options[:steps]
  total_steps = warmup_steps + timed_steps
  losses = []

  # Warm up the model (including one full timed-length pass) to avoid init overhead in measurements.
  step_idx = 0
  loader.each do |data, target|
    data = data.to(device)
    target = target.to(device)

    optimizer.zero_grad
    loss = Torch::NN::F.nll_loss(model.call(data), target)
    loss.backward
    optimizer.step

    step_idx += 1
    break if step_idx >= total_steps
  end

  sync_cuda_if_needed(device)
  Torch::Distributed.barrier if distributed

  timed = 0
  step_idx = 0
  start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  loader.each do |data, target|
    data = data.to(device)
    target = target.to(device)

    optimizer.zero_grad
    loss = Torch::NN::F.nll_loss(model.call(data), target)
    loss.backward
    optimizer.step

    loss_value = loss.item
    if distributed
      loss_tensor = Torch.tensor([loss_value], device: device)
      Torch::Distributed.all_reduce(loss_tensor)
      loss_value = loss_tensor.item / world_size.to_f
    end
    losses << loss_value if !distributed || rank.zero?

    step_idx += 1
    break if step_idx >= timed_steps
  end

  sync_cuda_if_needed(device)
  Torch::Distributed.barrier if distributed
  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
  timed = step_idx

  images = timed * options[:batch_size] * world_size
  throughput = elapsed.positive? ? images.to_f / elapsed : 0.0
  initial_loss = losses.first || 0.0
  final_loss = losses.last || initial_loss
  loss_delta = initial_loss - final_loss
  loss_delta_per_step = timed.zero? ? 0.0 : loss_delta / timed
  loss_delta_per_sec = elapsed.zero? ? 0.0 : loss_delta / elapsed

  result = if !distributed || rank.zero?
    {
      backend: selected_backend,
      world_size: world_size,
      batch_size: options[:batch_size],
      arch: arch,
      dataset: config[:dataset],
      elapsed: elapsed,
      timed_steps: timed,
      images: images,
      throughput: throughput,
      initial_loss: initial_loss,
      final_loss: final_loss,
      loss_delta: loss_delta,
      loss_delta_per_step: loss_delta_per_step,
      loss_delta_per_sec: loss_delta_per_sec
    }
  end

  Torch::Distributed.destroy_process_group if distributed
  result
end

def run_benchmark_case(world_size, options)
  if world_size > 1
    outputs = Torch::Distributed.fork_world(world_size, start_method: :spawn) do |rank, port|
      benchmark_worker(rank, world_size, port, options)
    end
    outputs.compact.first
  else
    benchmark_worker(0, 1, Torch::Distributed.free_port, options)
  end
end

def print_summary_table(results)
  puts "\nBenchmark comparison (processing vs convergence)"
  puts "Processing speed: images per second. Convergence speed: average loss reduction per step and per second.\n"

  headers = ["Backend", "Proc Group", "Batch", "Images/s", "Loss delta/step", "Loss delta/s", "Final loss"]
  formatters = [
    ->(r) { r[:backend] },
    ->(r) { r[:world_size] },
    ->(r) { r[:batch_size] },
    ->(r) { format("%.1f", r[:throughput]) },
    ->(r) { format("%.4f", r[:loss_delta_per_step]) },
    ->(r) { format("%.4f", r[:loss_delta_per_sec]) },
    ->(r) { format("%.4f", r[:final_loss]) }
  ]

  widths = headers.each_with_index.map do |header, idx|
    [header.length, results.map { |r| formatters[idx].call(r).to_s.length }.max].compact.max
  end

  header_line = headers.each_with_index.map { |h, idx| h.ljust(widths[idx]) }.join(" | ")
  divider = widths.map { |w| "-" * w }.join("-+-")
  puts header_line
  puts divider

  results.sort_by { |r| [r[:backend], r[:world_size], r[:batch_size]] }.each do |result|
    row = formatters.each_with_index.map { |formatter, idx| formatter.call(result).to_s.ljust(widths[idx]) }
    puts row.join(" | ")
  end
end

options = parse_options
apply_spawn_overrides!(options)
max_world_size = options[:gpus]
raise "Number of GPUs requested must be >= 1" if max_world_size < 1
Torch.manual_seed(1)

group_sizes = options[:group_sizes].map { |v| [v, max_world_size].min }.select { |v| v >= 1 }.uniq.sort
batch_sizes = options[:batch_sizes].map { |v| [v, 1].max }.uniq
backends = options[:backends].map(&:downcase).uniq

if group_sizes.any? { |size| size > 1 }
  raise "torch.distributed is not available" unless Torch::Distributed.available?
end

results = []

backends.each do |backend|
  unless backend_supported?(backend)
    warn "Skipping backend=#{backend} because required accelerator support is unavailable."
    next
  end

  group_sizes.each do |world_size|
    batch_sizes.each do |batch_size|
      run_options = options.merge(batch_size: batch_size, backend: backend, gpus: world_size)
      puts "Running backend=#{backend}, group_size=#{world_size}, batch_size=#{batch_size}..." unless spawn_worker_process?
      with_spawn_env(backend: backend, group_size: world_size, batch_size: batch_size) do
        results << run_benchmark_case(world_size, run_options)
      end
    end
  end
end

results.compact!

if results.empty?
  puts "No benchmark results to report."
else
  print_summary_table(results)
end
