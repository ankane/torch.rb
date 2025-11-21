require "socket"
require "rbconfig"

module Torch
  module Distributed
    DEFAULT_DEVICE_BACKENDS = {
      "cpu" => "gloo",
      "cuda" => "nccl",
      "xpu" => "xccl",
      "mps" => "gloo"
    }.freeze

    SPAWN_ENV_KEY = "TORCH_DISTRIBUTED_SPAWNED".freeze
    SPAWN_RANK_ENV_KEY = "TORCH_DISTRIBUTED_SPAWN_RANK".freeze
    SPAWN_WORLD_SIZE_ENV_KEY = "TORCH_DISTRIBUTED_SPAWN_WORLD_SIZE".freeze
    SPAWN_PORT_ENV_KEY = "TORCH_DISTRIBUTED_SPAWN_PORT".freeze
    SPAWN_PIPE_ENV_KEY = "TORCH_DISTRIBUTED_SPAWN_PIPE".freeze
    SPAWN_SCRIPT_ENV_KEY = "TORCH_DISTRIBUTED_SPAWN_SCRIPT".freeze
    SPAWN_TEST_ENV_KEY = "TORCH_DISTRIBUTED_SPAWN_TEST".freeze
    SPAWN_ARGV = ARGV.dup.freeze

    class << self
      def initialized?
        _initialized?
      end

      def init_process_group(backend = nil, init_method: "env://", store: nil, rank: nil, world_size: nil, timeout: DEFAULT_TIMEOUT, wait_for_workers: true, device_id: nil)
        raise Torch::Error, "torch.distributed is not available" unless available?

        backend ||= default_backend_for(device_id)

        if store.nil?
          case init_method
          when "env://"
            rank = Integer(ENV.fetch("RANK")) if rank.nil?
            world_size = Integer(ENV.fetch("WORLD_SIZE")) if world_size.nil?
            master_addr = ENV.fetch("MASTER_ADDR", "127.0.0.1")
            master_port = Integer(ENV.fetch("MASTER_PORT", "29500"))
            raise ArgumentError, "rank is required" if rank.nil?
            raise ArgumentError, "world_size is required" if world_size.nil?
            is_master = rank.zero?
            store = TCPStore.new(master_addr, master_port, world_size, is_master, wait_for_workers: wait_for_workers, timeout: timeout)
          else
            raise ArgumentError, "store is required when using init_method=#{init_method.inspect}"
          end
        end

        raise ArgumentError, "rank is required" if rank.nil?
        raise ArgumentError, "world_size is required" if world_size.nil?

        device_id ||= default_device_id_for_backend(backend, rank, world_size)

        timeout_ms = (timeout * 1000).to_i
        bound_device_id = device_id.nil? ? -1 : Integer(device_id)
        if backend == "nccl" && bound_device_id >= 0 && Torch.const_defined?(:CUDA) && Torch::CUDA.respond_to?(:set_device)
          device_count = Torch::CUDA.device_count if Torch::CUDA.respond_to?(:device_count)
          # Only attempt to switch devices when the requested id exists to avoid
          # raising on hosts with fewer GPUs than the provided local rank.
          Torch::CUDA.set_device(bound_device_id) if device_count.nil? || bound_device_id < device_count
        end
        pg = _init_process_group(backend, store, rank, world_size, timeout_ms, bound_device_id)
        warmup_process_group(pg, backend)
      end

      def destroy_process_group
        _destroy_process_group
      end

      def default_process_group
        _default_process_group
      end

      def get_world_size(group = nil)
        ensure_process_group!(group)
        _get_world_size(group)
      end

      def get_rank(group = nil)
        ensure_process_group!(group)
        _get_rank(group)
      end

      def barrier(group: nil)
        ensure_process_group!(group)
        _barrier(group)
      end

      def all_reduce(tensor, op: ReduceOp::SUM, group: nil)
        ensure_process_group!(group)
        _all_reduce(tensor, op, group)
      end

      def broadcast(tensor, src:, group: nil)
        ensure_process_group!(group)
        _broadcast(tensor, src, group)
      end

      def register_ddp_hook(tensor, process_group, world_size)
        ensure_process_group!(process_group)
        _register_ddp_hook(tensor, process_group, Integer(world_size))
      rescue NoMethodError
        # Fallback for environments built without the native helper; this may
        # still call back into Ruby from autograd threads.
        tensor.register_hook do |grad|
          all_reduce(grad, group: process_group)
          grad.div!(world_size.to_f)
        end
      end

      def get_default_backend_for_device(device)
        backend = DEFAULT_DEVICE_BACKENDS[device_type_from(device)]
        raise ArgumentError, "Default backend not registered for device: #{device.inspect}" unless backend
        backend
      end

      def fork_world(world_size, host: "127.0.0.1", start_method: :fork, &block)
        raise ArgumentError, "world_size must be positive" unless world_size.to_i.positive?
        raise ArgumentError, "block required" unless block

        start_method = normalize_start_method(start_method)
        return run_spawn_worker(&block) if start_method == :spawn && spawn_worker?

        fork_spawn_world(world_size, host: host, start_method: start_method, &block)
      end

      def fork_spawn_world(world_size, host:, start_method:, &block)
        port = free_port(host: host)
        readers = []
        pids = []
        pgid = nil
        completed = false

        begin
          world_size.times do |rank|
            reader, writer = IO.pipe
            begin
              case start_method
              when :fork
                pids << fork_worker(reader, writer, rank, port, world_size, &block)
              when :spawn
                pid, pgid = spawn_worker(reader, writer, rank, port, host: host, world_size: world_size, pgid: pgid)
                pids << pid
              else
                raise ArgumentError, "Unsupported start_method: #{start_method.inspect}"
              end
              readers << reader
              writer.close unless writer.closed?
            rescue Exception
              reader.close unless reader.closed?
              writer.close unless writer.closed?
              raise
            end
          end

          read_failure = Object.new

          outputs = readers.map do |reader|
            begin
              Marshal.load(reader)
            rescue EOFError
              read_failure
            ensure
              reader.close unless reader.closed?
            end
          end

          statuses = pids.each_with_index.map do |pid, idx|
            _pid, status = Process.wait2(pid)
            [idx, pid, status]
          end

          statuses.each do |idx, pid, status|
            output = outputs[idx]
            if output.equal?(read_failure)
              raise Torch::Error, "Child #{pid} closed pipe before sending result (status #{status.exitstatus})"
            end
            if !status.success? || (output.is_a?(Hash) && output[:error])
              message = if output.is_a?(Hash) && output[:error]
                "Child #{pid} failed: #{output[:error]}\n#{Array(output[:backtrace]).join("\n")}"
              else
                "Child #{pid} exited with status #{status.exitstatus}"
              end
              raise Torch::Error, message
            end
          end

          completed = true
          outputs
        ensure
          # Ensure child workers are cleaned up if an interrupt or error occurs.
          terminate_processes(pids, pgid: pgid) unless completed
        end
      end

      def free_port(host: "127.0.0.1")
        server = TCPServer.new(host, 0)
        port = server.addr[1]
        server.close
        port
      end

      private

      def ensure_process_group!(group)
        return if group || initialized?

        raise Torch::Error, "Default process group is not initialized"
      end

      def default_device_id_for_backend(backend, rank, world_size)
        return unless backend == "nccl"

        default_local_rank(rank, world_size)
      end

      def warmup_process_group(pg, backend)
        return pg unless backend == "nccl"

        # Only warm up when a native process group was returned.
        # Test helpers may stub out `_init_process_group` and return arbitrary
        # Ruby objects, which cannot be passed to the C++ bindings.
        return pg unless pg.nil? || (defined?(Torch::Distributed::ProcessGroup) && pg.is_a?(Torch::Distributed::ProcessGroup))

        # Prime NCCL communicators so the first user-visible collective is fast
        _barrier(pg)
        pg
      rescue
        _destroy_process_group
        raise
      end

      def default_local_rank(rank, world_size)
        local_rank = env_integer("LOCAL_RANK")
        return local_rank unless local_rank.nil?

        local_world_size = env_integer("LOCAL_WORLD_SIZE") || world_size
        return unless local_world_size && rank

        rank % local_world_size if local_world_size.positive?
      end

      def env_integer(key)
        Integer(ENV[key]) if ENV.key?(key)
      rescue ArgumentError
        nil
      end

      def default_backend_for(device_id)
        get_default_backend_for_device(device_id)
      end

      def device_type_from(device)
        case device
        when Torch::Device
          device.type
        when NilClass
          accelerator_type || "cpu"
        when String
          Torch.device(device).type
        when Integer
          return accelerator_type || "cpu" if device.negative?
          if Torch.const_defined?(:CUDA) && Torch::CUDA.respond_to?(:device_count)
            max = Torch::CUDA.device_count
            return accelerator_type || "cpu" if max <= 0 || device >= max
            return Torch.device("cuda:#{device}").type
          end
          accelerator_type || "cpu"
        else
          return device.type if device.respond_to?(:type)
          Torch.device(device).type
        end
      rescue => e
        raise ArgumentError, "Invalid device #{device.inspect}: #{e.message}"
      end

      def accelerator_type
        acc = Torch::Accelerator.current_accelerator
        acc.type if acc && acc.respond_to?(:type)
      rescue
        nil
      end

      def normalize_start_method(start_method)
        method = start_method&.to_sym
        return method if [:fork, :spawn].include?(method)

        raise ArgumentError, "start_method must be :fork or :spawn (got #{start_method.inspect})"
      end

      def spawn_worker?
        ENV[SPAWN_ENV_KEY] == "1"
      end

      def run_spawn_worker(&block)
        rank = Integer(ENV.fetch(SPAWN_RANK_ENV_KEY))
        port = Integer(ENV.fetch(SPAWN_PORT_ENV_KEY))
        pipe_fd = Integer(ENV.fetch(SPAWN_PIPE_ENV_KEY))

        writer = IO.new(pipe_fd, "wb")
        writer.binmode
        writer.sync = true

        result = block.call(rank, port)
        Marshal.dump(result, writer)
        writer.flush
        writer.close
        Process.exit!(0)
      rescue Exception => e
        begin
          if defined?(writer) && writer && !writer.closed?
            Marshal.dump({error: "#{e.class}: #{e.message}", backtrace: e.backtrace}, writer)
            writer.flush
            writer.close
          end
        rescue StandardError
          # best-effort error reporting back to parent
        ensure
          Process.exit!(1)
          end
      end

      def fork_worker(reader, writer, rank, port, world_size, &block)
        fork do
          reader.close
          begin
            ENV["LOCAL_RANK"] = rank.to_s
            ENV["LOCAL_WORLD_SIZE"] = world_size.to_s
            ENV["RANK"] = rank.to_s
            ENV["WORLD_SIZE"] = world_size.to_s
            writer.binmode
            writer.sync = true
            result = block.call(rank, port)
            Marshal.dump(result, writer)
            writer.flush
            writer.close
            Process.exit!(0)
          rescue => e
            Marshal.dump({error: "#{e.class}: #{e.message}", backtrace: e.backtrace}, writer)
            writer.flush
            writer.close
            Process.exit!(1)
          ensure
            writer.close unless writer.closed?
          end
        end
      end

      def spawn_worker(reader, writer, rank, port, host:, world_size:, pgid: nil)
        writer.binmode
        writer.close_on_exec = false

        script = ENV[SPAWN_SCRIPT_ENV_KEY] || $0
        env = {
          SPAWN_ENV_KEY => "1",
          SPAWN_RANK_ENV_KEY => rank.to_s,
          SPAWN_WORLD_SIZE_ENV_KEY => world_size.to_s,
          SPAWN_PORT_ENV_KEY => port.to_s,
          SPAWN_PIPE_ENV_KEY => writer.fileno.to_s,
          "LOCAL_RANK" => rank.to_s,
          "LOCAL_WORLD_SIZE" => world_size.to_s,
          "MASTER_ADDR" => host,
          "MASTER_PORT" => port.to_s,
          "RANK" => rank.to_s,
          "WORLD_SIZE" => world_size.to_s
        }
        env["RUBYLIB"] = [ENV["RUBYLIB"], $LOAD_PATH.join(File::PATH_SEPARATOR)].compact.reject(&:empty?).join(File::PATH_SEPARATOR)

        spawn_opts = {close_others: false}
        spawn_opts[:pgroup] = pgid ? pgid : true

        pid = Process.spawn(env, RbConfig.ruby, script, *spawn_argv, spawn_opts)
        pgid ||= pid
        [pid, pgid]
      rescue SystemCallError => e
        raise Torch::Error, "failed to spawn worker #{rank}: #{e.message}"
      end

      def spawn_argv
        test_filter = ENV[SPAWN_TEST_ENV_KEY]
        return SPAWN_ARGV unless test_filter
        return SPAWN_ARGV if SPAWN_ARGV.include?("-n")

        # Restrict child to the specific test that triggered the spawn
        SPAWN_ARGV + ["-n", test_filter]
      end

      def terminate_processes(pids, pgid: nil)
        return if pids.empty? && !pgid

        send_process_group_signal(pgid, "TERM")
        pids.each { |pid| safe_kill(pid, "TERM") }
        sleep(0.2)
        pids.each do |pid|
          next unless process_alive?(pid)

          safe_kill(pid, "KILL")
        end
        pids.each do |pid|
          begin
            Process.wait(pid)
          rescue Errno::ECHILD
          end
        end
      end

      def send_process_group_signal(pgid, sig)
        return unless pgid

        Process.kill(sig, -pgid)
      rescue Errno::ESRCH
      end

      def safe_kill(pid, sig)
        Process.kill(sig, pid)
      rescue Errno::ESRCH
      end

      def process_alive?(pid)
        Process.kill(0, pid)
        true
      rescue Errno::ESRCH
        false
      end
    end

    class TCPStore
      def self.new(host, port, world_size, is_master, wait_for_workers: true, timeout: DEFAULT_TIMEOUT)
        Torch::Distributed._create_tcp_store(host, port, world_size, is_master, (timeout * 1000).to_i, wait_for_workers)
      end
    end

    class FileStore
      def self.new(path, world_size)
        Torch::Distributed._create_file_store(path, world_size)
      end
    end

    if respond_to?(:_create_hash_store)
      class HashStore
        def self.new
          Torch::Distributed._create_hash_store
        end
      end
    end
  end
end

at_exit do
  begin
    Torch::Distributed.destroy_process_group if Torch::Distributed.available? && Torch::Distributed.initialized?
  rescue Exception
    # best-effort cleanup to avoid leaked process groups
  end
end
