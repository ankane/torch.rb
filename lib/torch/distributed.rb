require "socket"

module Torch
  module Distributed
    DEFAULT_DEVICE_BACKENDS = {
      "cpu" => "gloo",
      "cuda" => "nccl",
      "xpu" => "xccl",
      "mps" => "gloo"
    }.freeze

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

        timeout_ms = (timeout * 1000).to_i
        _init_process_group(backend, store, rank, world_size, timeout_ms)
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

      def get_default_backend_for_device(device)
        backend = DEFAULT_DEVICE_BACKENDS[device_type_from(device)]
        raise ArgumentError, "Default backend not registered for device: #{device.inspect}" unless backend
        backend
      end

      def fork_world(world_size, host: "127.0.0.1")
        raise ArgumentError, "world_size must be positive" unless world_size.to_i.positive?
        raise ArgumentError, "block required" unless block_given?

        port = free_port(host: host)
        readers = []
        pids = []
        world_size.times do |rank|
          reader, writer = IO.pipe
          pid = fork do
            reader.close
            begin
              writer.binmode
              result = yield(rank, port)
              Marshal.dump(result, writer)
              exit! 0
            rescue => e
              Marshal.dump({error: "#{e.class}: #{e.message}", backtrace: e.backtrace}, writer)
              exit! 1
            ensure
              writer.close unless writer.closed?
            end
          end
          writer.close
          readers << reader
          pids << pid
        end

        outputs = readers.map do |reader|
          data = Marshal.load(reader)
          reader.close
          data
        end

        statuses = pids.each_with_index.map do |pid, idx|
          _pid, status = Process.wait2(pid)
          [idx, pid, status]
        end

        statuses.each do |idx, pid, status|
          output = outputs[idx]
          if !status.success? || (output.is_a?(Hash) && output[:error])
            message = if output.is_a?(Hash) && output[:error]
              "Child #{pid} failed: #{output[:error]}\n#{Array(output[:backtrace]).join("\n")}"
            else
              "Child #{pid} exited with status #{status.exitstatus}"
            end
            raise Torch::Error, message
          end
        end

        outputs
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

      def default_backend_for(device_id)
        get_default_backend_for_device(device_id)
      end

      def device_type_from(device)
        case device
        when Torch::Device
          device.type
        when String
          Torch.device(device).type
        when Integer
          Torch.device("cuda:#{device}").type
        when NilClass
          Torch::Accelerator.current_accelerator&.type || "cpu"
        else
          Torch.device(device).type
        end
      rescue => e
        raise ArgumentError, "Invalid device #{device.inspect}: #{e.message}"
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
