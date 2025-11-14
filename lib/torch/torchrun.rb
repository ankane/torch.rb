# frozen_string_literal: true

require "optparse"
require "socket"
require "etc"
require "securerandom"
require "rbconfig"

require_relative "../torch"

module Torch
  module TorchRun
    SIGNALS = %w[INT TERM QUIT].freeze

    class Error < StandardError; end

    class Parser
      attr_reader :parser

      def initialize
        @parser = OptionParser.new
      end

      def parse(argv)
        options = default_options

        parser.banner = "Usage: torchrun [options] TRAINING_SCRIPT [script args]"
        parser.separator ""
        parser.separator "Launch parameters:"

        parser.on("--nnodes MIN[:MAX]", String, "Number of nodes or range (default: #{options[:nnodes]})") do |value|
          options[:nnodes] = value
        end

        parser.on("--nproc-per-node VALUE", String, "Processes per node (int, gpu, cpu, auto). Default: #{options[:nproc_per_node]}") do |value|
          options[:nproc_per_node] = value
        end

        parser.on("--node-rank VALUE", Integer, "Rank of the node for multi-node jobs. Default: #{options[:node_rank]}") do |value|
          options[:node_rank] = value
        end

        parser.on("--rdzv-backend NAME", String, "Rendezvous backend (static or c10d). Default: #{options[:rdzv_backend]}") do |value|
          options[:rdzv_backend] = value
        end

        parser.on("--rdzv-endpoint HOST[:PORT]", String, "Rendezvous endpoint. Default: use --master-addr/--master-port") do |value|
          options[:rdzv_endpoint] = value
        end

        parser.on("--rdzv-id ID", String, "User defined job id. Default: #{options[:rdzv_id]}") do |value|
          options[:rdzv_id] = value
        end

        parser.on("--rdzv-conf CONF", String, "Additional rendezvous config (k=v,k2=v2)") do |value|
          options[:rdzv_conf] = parse_kv_pairs(value)
        end

        parser.on("--standalone", "Start a local rendezvous store on a free port") do
          options[:standalone] = true
        end

        parser.on("--max-restarts VALUE", Integer, "Restarts before failing. Default: #{options[:max_restarts]}") do |value|
          options[:max_restarts] = value
        end

        parser.on("--monitor-interval SECONDS", Float, "Delay between restart attempts. Default: #{options[:monitor_interval]}") do |value|
          options[:monitor_interval] = value
        end

        parser.on("--role NAME", String, "Role for the worker group. Default: #{options[:role]}") do |value|
          options[:role] = value
        end

        parser.on("--master-addr HOST", String, "Master address for static rendezvous. Default: #{options[:master_addr]}") do |value|
          options[:master_addr] = value
        end

        parser.on("--master-port PORT", Integer, "Master port for static rendezvous. Default: #{options[:master_port]}") do |value|
          options[:master_port] = value
        end

        parser.on("--pass-local-rank-arg", "Append --local-rank to the training script invocation") do
          options[:pass_local_rank_arg] = true
        end

        parser.on("--no-ruby", "Execute the training script directly instead of `#{RbConfig.ruby}`") do
          options[:no_ruby] = true
        end

        parser.on("-h", "--help", "Prints this help") do
          puts parser
          exit
        end

        rest = parser.parse!(argv)
        raise OptionParser::MissingArgument, "training_script" if rest.empty?

        training_script = rest.shift
        [options, training_script, rest]
      end

      def to_s
        parser.to_s
      end

      private

      def default_options
        {
          nnodes: "1:1",
          nproc_per_node: "1",
          node_rank: 0,
          rdzv_backend: "static",
          rdzv_endpoint: "",
          rdzv_id: "none",
          rdzv_conf: {},
          standalone: false,
          max_restarts: 0,
          monitor_interval: 1.0,
          role: "default",
          master_addr: "127.0.0.1",
          master_port: 29_500,
          pass_local_rank_arg: false,
          no_ruby: false
        }
      end

      def parse_kv_pairs(value)
        return {} if value.nil? || value.strip.empty?

        value.split(",").each_with_object({}) do |pair, acc|
          key, val = pair.split("=", 2)
          raise OptionParser::InvalidArgument, "Invalid rendezvous config entry: #{pair.inspect}" unless key && val

          acc[key.strip] = val.strip
        end
      end
    end

    module_function

    def start(argv, out: $stdout, err: $stderr)
      parser = Parser.new
      options, script, script_args = parser.parse(argv)
      status = Launcher.new(options, script, script_args, out: out, err: err).run
      exit(status)
    rescue OptionParser::ParseError => e
      err.puts(e.message)
      err.puts(parser)
      exit(2)
    rescue Error => e
      err.puts("torchrun: #{e.message}")
      exit(1)
    end

    class Launcher
      def initialize(options, script, script_args, out: $stdout, err: $stderr)
        @options = options
        @script = script
        @script_args = script_args
        @out = out
        @err = err

        @local_world_size = determine_local_world_size(@options[:nproc_per_node])
        @min_nodes, @max_nodes = parse_nnodes(@options[:nnodes])
        @num_nodes = ensure_fixed_nnodes(@min_nodes, @max_nodes)
        @node_rank = @options[:node_rank]
        @max_restarts = [@options[:max_restarts], 0].max
        @monitor_interval = [@options[:monitor_interval], 0.0].max
        @role = @options[:role]
        @pass_local_rank_arg = @options[:pass_local_rank_arg]
        @no_ruby = @options[:no_ruby]
        validate_node_rank!

        setup_rendezvous!
      end

      def run
        restarts = 0

        loop do
          status = launch_worker_group(restarts)
          return status if status.zero? || @signal_received
          return status if restarts >= @max_restarts

          restarts += 1
          log("Worker group failed (exit #{status}). Restarting #{restarts}/#{@max_restarts} ...")
          sleep(@monitor_interval) if @monitor_interval.positive?
        end
      end

      private

      def launch_worker_group(restart_count)
        @signal_received = nil
        @current_pids = spawn_workers(restart_count)
        handler_state = setup_signal_handlers
        status = monitor_workers(@current_pids.dup)
        cleanup_workers(@current_pids)
        restore_signal_handlers(handler_state)
        return signal_exit_status if @signal_received

        status
      ensure
        @current_pids = []
      end

      def spawn_workers(restart_count)
        base_env = base_environment(restart_count)
        Array.new(@local_world_size) do |local_rank|
          env = base_env.merge(rank_environment(local_rank))
          spawn_worker(env, local_rank)
        end
      end

      def spawn_worker(env, local_rank)
        args = command_arguments(local_rank)
        Process.spawn(env, *args)
      rescue SystemCallError => e
        raise Error, "failed to launch worker #{local_rank}: #{e.message}"
      end

      def command_arguments(local_rank)
        cmd = []
        if @no_ruby
          cmd << @script
        else
          cmd << RbConfig.ruby
          cmd << @script
        end
        cmd.concat(@script_args)
        cmd << "--local-rank=#{local_rank}" if @pass_local_rank_arg
        cmd
      end

      def base_environment(restart_count)
        endpoint = "#{@master_addr}:#{@master_port}"
        env = {
          "MASTER_ADDR" => @master_addr,
          "MASTER_PORT" => @master_port.to_s,
          "WORLD_SIZE" => world_size.to_s,
          "LOCAL_WORLD_SIZE" => @local_world_size.to_s,
          "GROUP_RANK" => @node_rank.to_s,
          "TORCHRUN_ROLE" => @role,
          "TORCHRUN_NNODES" => @num_nodes.to_s,
          "TORCHRUN_NPROC_PER_NODE" => @local_world_size.to_s,
          "TORCHELASTIC_RUN_ID" => @rdzv_id,
          "TORCHRUN_RDZV_BACKEND" => @rdzv_backend,
          "TORCHRUN_RDZV_ENDPOINT" => endpoint,
          "TORCHELASTIC_RESTART_COUNT" => restart_count.to_s,
          "TORCHRUN_STANDALONE" => @standalone ? "1" : "0"
        }
        unless @rdzv_conf.empty?
          env["TORCHRUN_RDZV_CONF"] = @rdzv_conf.map { |k, v| "#{k}=#{v}" }.join(",")
        end
        ENV.to_h.merge(env)
      end

      def rank_environment(local_rank)
        rank = @node_rank * @local_world_size + local_rank
        {
          "LOCAL_RANK" => local_rank.to_s,
          "RANK" => rank.to_s
        }
      end

      def monitor_workers(pids)
        exit_code = 0
        remaining = pids.dup
        until remaining.empty?
          pid, status = Process.wait2
          next unless pid

          remaining.delete(pid)
          unless status.success?
            exit_code = exit_status_from(status)
            terminate_workers(remaining)
            break
          end
        end
        exit_code
      rescue Errno::ECHILD
        0
      end

      def terminate_workers(pids)
        return if pids.empty?

        pids.each { |pid| send_signal(pid, "TERM") }
        sleep(0.2)
        pids.each do |pid|
          next unless process_alive?(pid)

          send_signal(pid, "KILL")
        end
        pids.each do |pid|
          begin
            Process.wait(pid)
          rescue Errno::ECHILD
          end
        end
      end

      def process_alive?(pid)
        Process.kill(0, pid)
        true
      rescue Errno::ESRCH
        false
      end

      def setup_signal_handlers
        SIGNALS.each_with_object({}) do |sig, acc|
          next unless Signal.list.key?(sig)

          previous = Signal.trap(sig) do
            @signal_received = sig
            forward_signal(sig)
          end
          acc[sig] = previous
        end
      end

      def forward_signal(sig)
        (@current_pids || []).each { |pid| send_signal(pid, sig) }
      end

      def restore_signal_handlers(state)
        return unless state

        state.each do |sig, previous|
          Signal.trap(sig, previous)
        end
      end

      def send_signal(pid, sig)
        Process.kill(sig, pid)
      rescue Errno::ESRCH
        nil
      end

      def cleanup_workers(pids)
        pids.each do |pid|
          next unless process_alive?(pid)

          begin
            Process.wait(pid)
          rescue Errno::ECHILD
          end
        end
      end

      def signal_exit_status
        return 0 unless @signal_received

        128 + Signal.list.fetch(@signal_received, 0)
      end

      def exit_status_from(status)
        if status.exited?
          status.exitstatus
        elsif status.signaled?
          128 + status.termsig
        else
          1
        end
      end

      def determine_local_world_size(value)
        spec = value.to_s.strip.downcase
        case spec
        when "", "1"
          1
        when /\A\d+\z/
          amount = spec.to_i
          raise Error, "nproc-per-node must be >= 1" if amount < 1

          amount
        when "gpu"
          gpu_count = cuda_device_count
          raise Error, "CUDA is not available for --nproc-per-node=gpu" if gpu_count.zero?

          gpu_count
        when "auto"
          gpu_count = cuda_device_count
          return gpu_count if gpu_count.positive?

          cpu_count
        when "cpu"
          cpu_count
        else
          raise Error, "Unsupported --nproc-per-node value: #{value}"
        end
      end

      def cuda_device_count
        return 0 unless defined?(Torch::CUDA)
        return 0 unless Torch::CUDA.respond_to?(:available?) && Torch::CUDA.available?
        return 0 unless Torch::CUDA.respond_to?(:device_count)

        Torch::CUDA.device_count
      rescue StandardError
        0
      end

      def cpu_count
        Etc.respond_to?(:nprocessors) ? (Etc.nprocessors || 1) : 1
      rescue StandardError
        1
      end

      def parse_nnodes(value)
        parts = value.split(":")
        nums = parts.map do |part|
          Integer(part, exception: false)
        end
        raise Error, "Invalid --nnodes value: #{value.inspect}" if nums.any?(&:nil?)

        if nums.length == 1
          [nums.first, nums.first]
        elsif nums.length == 2
          [nums.first, nums.last]
        else
          raise Error, "Invalid --nnodes value: #{value.inspect}"
        end
      end

      def ensure_fixed_nnodes(min_nodes, max_nodes)
        raise Error, "--nnodes minimum must be >= 1" if min_nodes < 1
        raise Error, "--nnodes maximum must be >= minimum" if max_nodes < min_nodes
        raise Error, "Elastic nnodes ranges are not supported yet (got #{min_nodes}:#{max_nodes})" if min_nodes != max_nodes

        min_nodes
      end

      def world_size
        @world_size ||= @num_nodes * @local_world_size
      end

      def validate_node_rank!
        raise Error, "--node-rank must be >= 0" if @node_rank.negative?
        raise Error, "--node-rank (#{@node_rank}) must be less than --nnodes (#{@num_nodes})" if @node_rank >= @num_nodes
      end

      def setup_rendezvous!
        @rdzv_backend = normalize_backend(@options[:rdzv_backend])
        @rdzv_conf = @options[:rdzv_conf] || {}
        if @options[:standalone]
          configure_standalone_rendezvous
        else
          configure_static_rendezvous
        end
      end

      def normalize_backend(value)
        backend = value.to_s.downcase
        raise Error, "Unsupported rendezvous backend: #{value.inspect}" unless %w[static c10d].include?(backend)

        backend
      end

      def configure_standalone_rendezvous
        @standalone = true
        @rdzv_backend = "c10d"
        @rdzv_id = SecureRandom.uuid
        @master_addr = "127.0.0.1"
        @master_port = find_free_port(@master_addr)
        log(<<~MSG)

          **************************************
          Rendezvous info:
          --rdzv-backend=#{@rdzv_backend}
          --rdzv-endpoint=#{@master_addr}:#{@master_port}
          --rdzv-id=#{@rdzv_id}
          **************************************

        MSG
      end

      def configure_static_rendezvous
        @standalone = false
        endpoint_host, endpoint_port = parse_endpoint(@options[:rdzv_endpoint])
        @master_addr = endpoint_host || @options[:master_addr]
        @master_port = endpoint_port || @options[:master_port]
        @rdzv_id = @options[:rdzv_id]
        raise Error, "MASTER_ADDR must be provided" if @master_addr.to_s.empty?
        raise Error, "MASTER_PORT must be > 0" unless @master_port.to_i.positive?
      end

      def parse_endpoint(value)
        return [nil, nil] if value.nil? || value.strip.empty?

        host, port_str = value.split(":", 2)
        port = port_str ? Integer(port_str, exception: false) : nil
        raise Error, "Invalid rendezvous endpoint: #{value.inspect}" if host.to_s.empty? || (port_str && port.nil?)

        [host, port]
      end

      def find_free_port(host)
        server = TCPServer.new(host, 0)
        server.addr[1]
      ensure
        server&.close
      end

      def log(message)
        @out.puts(message)
      end
    end
  end
end
