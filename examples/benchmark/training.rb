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
    batch_size: 128,
    steps: 100,
    warmup: 10,
    backend: DEFAULT_BACKEND,
    gpus: Torch::CUDA.available? ? [Torch::CUDA.device_count, 1].max : 1,
    data_dir: File.join(__dir__, "data"),
    lr: 0.01
  }

  OptionParser.new do |opts|
    opts.banner = "Usage: ruby examples/benchmark/training.rb [options]"
    opts.on("--arch NAME", "Architecture to benchmark (#{ARCH_CONFIGS.keys.join(', ')}, default: #{defaults[:arch]})") { |v| defaults[:arch] = v }
    opts.on("--batch-size N", Integer, "Batch size per process (default: #{defaults[:batch_size]})") { |v| defaults[:batch_size] = v }
    opts.on("--steps N", Integer, "Number of timed training steps (default: #{defaults[:steps]})") { |v| defaults[:steps] = v }
    opts.on("--warmup N", Integer, "Number of warmup steps not included in timing (default: #{defaults[:warmup]})") { |v| defaults[:warmup] = v }
    opts.on("--backend NAME", String, "Process group backend (default: #{defaults[:backend]})") { |v| defaults[:backend] = v }
    opts.on("--gpus N", Integer, "Number of GPUs/processes to use (1 for non-distributed)") { |v| defaults[:gpus] = v }
    opts.on("--data-dir PATH", String, "Directory for cached datasets (default: #{defaults[:data_dir]})") { |v| defaults[:data_dir] = v }
    opts.on("--lr FLOAT", Float, "Learning rate (default: #{defaults[:lr]})") { |v| defaults[:lr] = v }
  end.parse!(ARGV)

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
  if distributed
    store = Torch::Distributed::TCPStore.new("127.0.0.1", port, world_size, rank.zero?)
    accelerator = Torch::Accelerator.current_accelerator
    backend = options[:backend] || Torch::Distributed.get_default_backend_for_device(accelerator) || DEFAULT_BACKEND
    Torch::Distributed.init_process_group(backend, store: store, rank: rank, world_size: world_size)
  end

  device = if Torch::CUDA.available? && options[:gpus] > 0
    Torch.device("cuda:#{rank % Torch::CUDA.device_count}")
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

    step_idx += 1
    break if step_idx >= timed_steps
  end

  sync_cuda_if_needed(device)
  Torch::Distributed.barrier if distributed
  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
  timed = step_idx

  if rank.zero?
    images = timed * options[:batch_size] * world_size
    puts "Architecture: #{arch}"
    puts "Dataset: #{config[:dataset]}"
    puts "GPUs: #{world_size}"
    puts "Batch size per process: #{options[:batch_size]}"
    puts "Timed steps: #{timed}"
    puts "Total images: #{images}"
    puts format("Elapsed: %.3fs | Throughput: %.1f images/s", elapsed, images / elapsed)
  end

  Torch::Distributed.destroy_process_group if distributed
end

options = parse_options
world_size = options[:gpus]
raise "Number of GPUs requested must be >= 1" if world_size < 1
Torch.manual_seed(1)

if world_size > 1
  raise "torch.distributed is not available" unless Torch::Distributed.available?
  Torch::Distributed.fork_world(world_size, start_method: :spawn) do |rank, port|
    benchmark_worker(rank, world_size, port, options)
  end
else
  benchmark_worker(0, 1, Torch::Distributed.free_port, options)
end
