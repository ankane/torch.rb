# Distributed MNIST training with Torch::Distributed + DistributedDataParallel
# Run with: ruby examples/mnist/distributed.rb --gpus 2

require "bundler/setup"
require "optparse"
require "torch"
require "torchvision"
require "tmpdir"

unless Torch::Distributed.available?
  abort "torch.distributed was not built in this binary"
end

DEFAULT_CHECKPOINT_PATH = File.join(Dir.tmpdir, "mnist_ddp_checkpoint.pt")

class MyNet < Torch::NN::Module
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

def parse_options
  defaults = {
    epochs: 5,
    batch_size: 64,
    lr: 1.0,
    gamma: 0.7,
    backend: "gloo",
    gpus: Torch::CUDA.available? ? [Torch::CUDA.device_count, 1].max : 1,
    log_interval: 20,
    data_dir: File.join(__dir__, "data"),
    checkpoint_path: DEFAULT_CHECKPOINT_PATH,
    resume: false
  }

  OptionParser.new do |opts|
    opts.banner = "Usage: ruby distributed.rb [options]"
    opts.on("--epochs N", Integer, "Number of epochs (default: #{defaults[:epochs]})") { |v| defaults[:epochs] = v }
    opts.on("--batch-size N", Integer, "Batch size per process (default: #{defaults[:batch_size]})") { |v| defaults[:batch_size] = v }
    opts.on("--lr FLOAT", Float, "Learning rate (default: #{defaults[:lr]})") { |v| defaults[:lr] = v }
    opts.on("--gamma FLOAT", Float, "LR scheduler gamma (default: #{defaults[:gamma]})") { |v| defaults[:gamma] = v }
    opts.on("--backend NAME", String, "Process group backend (default: #{defaults[:backend]})") { |v| defaults[:backend] = v }
    opts.on("--gpus N", Integer, "Number of GPUs/processes to use") { |v| defaults[:gpus] = v }
    opts.on("--log-interval N", Integer, "Batches between log statements") { |v| defaults[:log_interval] = v }
    opts.on("--data-dir PATH", String, "Directory for cached MNIST data") { |v| defaults[:data_dir] = v }
    opts.on("--checkpoint PATH", String, "Checkpoint file to save to (default: #{defaults[:checkpoint_path]})") { |v| defaults[:checkpoint_path] = v }
    opts.on("--resume", "Load checkpoint weights before training if the file exists") { defaults[:resume] = true }
  end.parse!(ARGV)

  defaults
end

def load_datasets(rank, data_dir)
  transforms = TorchVision::Transforms::Compose.new([
    TorchVision::Transforms::ToTensor.new,
    TorchVision::Transforms::Normalize.new([0.1307], [0.3081])
  ])

  if rank.zero?
    train = TorchVision::Datasets::MNIST.new(data_dir, train: true, download: true, transform: transforms)
    test = TorchVision::Datasets::MNIST.new(data_dir, train: false, download: true, transform: transforms)
    Torch::Distributed.barrier
  else
    Torch::Distributed.barrier
    train = TorchVision::Datasets::MNIST.new(data_dir, train: true, download: false, transform: transforms)
    test = TorchVision::Datasets::MNIST.new(data_dir, train: false, download: false, transform: transforms)
  end

  [train, test]
end

def subset_for_rank(dataset, rank, world_size)
  indices = rank.step(dataset.size - 1, world_size).to_a
  Torch::Utils::Data::Subset.new(dataset, indices)
end

def checkpoint_map_location(device, rank)
  accelerator_device = Torch::Accelerator.current_accelerator
  return nil unless accelerator_device

  accelerator_type = accelerator_device.type
  target_index = device.index
  if target_index.nil? && Torch::Accelerator.respond_to?(:device_count)
    count = Torch::Accelerator.device_count
    target_index = count.positive? ? rank % count : 0
  end
  { "#{accelerator_type}:0" => "#{accelerator_type}:#{target_index}" }
end

def load_checkpoint_if_present(ddp, device, rank, path)
  return false unless path && File.exist?(path)

  Torch::Distributed.barrier
  kwargs = { weights_only: true }
  map_location = checkpoint_map_location(device, rank)
  kwargs[:map_location] = map_location if map_location
  state_dict = Torch.load(path, **kwargs)
  ddp.module.load_state_dict(state_dict)
  true
end

def save_checkpoint(ddp, path, rank)
  return unless path

  Torch.save(ddp.module.state_dict, path) if rank.zero?
  Torch::Distributed.barrier
  puts "Saved checkpoint to #{path}" if rank.zero?
end

def train_epoch(model, device, loader, optimizer, epoch, rank, log_interval)
  model.train
  loader.each_with_index do |(data, target), batch_idx|
    data = data.to(device)
    target = target.to(device)

    optimizer.zero_grad
    loss = Torch::NN::F.nll_loss(model.call(data), target)
    loss.backward
    optimizer.step

    next unless rank.zero? && (batch_idx % log_interval).zero?

    processed = batch_idx * data.size(0)
    total = loader.dataset.size
    percent = 100.0 * processed / total
    puts "Rank #{rank} | Epoch #{epoch} [#{processed}/#{total} (#{percent.round})%] Loss: #{'%.4f' % loss.item}"
  end
end

def evaluate(model, device, loader)
  model.eval
  loss = 0.0
  correct = 0
  Torch.no_grad do
    loader.each do |data, target|
      data = data.to(device)
      target = target.to(device)
      output = model.call(data)
      loss += Torch::NN::F.nll_loss(output, target, reduction: "sum").item
      pred = output.argmax(1, keepdim: true)
      correct += pred.eq(target.view_as(pred)).sum.item
    end
  end

  loss /= loader.dataset.size
  acc = 100.0 * correct / loader.dataset.size
  puts "Test set: Average loss: #{format('%.4f', loss)}, Accuracy: #{correct}/#{loader.dataset.size} (#{format('%.1f', acc)}%)"
end

def run_worker(rank, world_size, port, options)
  store = Torch::Distributed::TCPStore.new("127.0.0.1", port, world_size, rank.zero?)
  accelerator = Torch::Accelerator.current_accelerator
  backend = options[:backend] || Torch::Distributed.get_default_backend_for_device(accelerator)
  Torch::Distributed.init_process_group(backend, store: store, rank: rank, world_size: world_size)

  device = if Torch::CUDA.available? && options[:gpus] > 0
    Torch.device("cuda:#{rank % Torch::CUDA.device_count}")
  else
    Torch.device("cpu")
  end

  model = MyNet.new.to(device)
  ddp = Torch::NN::Parallel::DistributedDataParallel.new(model, device_ids: device.type == "cuda" ? [device.index] : nil)
  optimizer = Torch::Optim::Adadelta.new(ddp.module.parameters, lr: options[:lr])
  scheduler = Torch::Optim::LRScheduler::StepLR.new(optimizer, step_size: 1, gamma: options[:gamma])

  train_dataset, test_dataset = load_datasets(rank, options[:data_dir])
  train_subset = subset_for_rank(train_dataset, rank, world_size)
  train_loader = Torch::Utils::Data::DataLoader.new(train_subset, batch_size: options[:batch_size], shuffle: true)
  test_loader = Torch::Utils::Data::DataLoader.new(test_dataset, batch_size: options[:batch_size], shuffle: false) if rank.zero?
  checkpoint_path = options[:checkpoint_path]

  if options[:resume]
    loaded = load_checkpoint_if_present(ddp, device, rank, checkpoint_path)
    if rank.zero?
      if loaded
        puts "Loaded checkpoint weights from #{checkpoint_path}"
      else
        puts "No checkpoint found at #{checkpoint_path}, starting from random initialization"
      end
    end
  end

  options[:epochs].times do |epoch_idx|
    epoch = epoch_idx + 1
    train_epoch(ddp, device, train_loader, optimizer, epoch, rank, options[:log_interval])
    if rank.zero?
      evaluate(ddp.module, device, test_loader)
    end
    save_checkpoint(ddp, checkpoint_path, rank) if checkpoint_path
  end

  Torch::Distributed.destroy_process_group
end

options = parse_options
world_size = options[:gpus]
raise "Number of GPUs requested must be >= 1" if world_size < 1
if Torch::CUDA.available?
  max_devices = Torch::CUDA.device_count
  if world_size > max_devices
    raise "Requested #{world_size} GPUs but only #{max_devices} visible"
  end
else
  puts "CUDA not available, running #{world_size} CPU workers"
end

Torch.manual_seed(1)

if world_size == 1
  run_worker(0, 1, Torch::Distributed.free_port, options)
else
  Torch::Distributed.fork_world(world_size) do |rank, port|
    run_worker(rank, world_size, port, options)
  end
end
