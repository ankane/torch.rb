module Torch
  module NN
    module Parallel
      # Implements data parallelism at the module level.
      #
      # This container parallelizes the application of the given module by
      # splitting the input across the specified devices by chunking in the
      # batch dimension. In the forward pass, the module is replicated on each
      # device, and each replica handles a portion of the input. During the
      # backwards pass, gradients from each replica are summed into the
      # original module.
      #
      # @note Backward Pass for Models Returning Loss
      #   When your model returns a scalar loss (e.g., GPT models that return
      #   [logits, loss]), you must use the {#backward} method instead of
      #   calling loss.backward directly. This is because gathering scalar
      #   tensors across devices breaks the autograd graph in torch.rb.
      #
      #   @example Training loop with loss-returning model
      #     dp_model = Torch::NN::DataParallel.new(model, device_ids: [0, 1])
      #     optimizer.zero_grad
      #     logits, loss = dp_model.call(input, targets: targets)
      #     dp_model.backward(scale: 1.0 / gradient_accumulation_steps)
      #     optimizer.step
      #
      # @example Basic usage (model returns output only)
      #   model = MyModel.new.to("cuda:0")
      #   parallel_model = Torch::NN::DataParallel.new(model, device_ids: [0, 1])
      #   output = parallel_model.call(input)
      #   loss = criterion.call(output, target)
      #   loss.backward  # Standard backward works when loss computed after gather
      #
      class DataParallel < Module
        attr_reader :module, :device_ids, :output_device, :dim
        alias_method :wrapped_module, :module

        # @param mod [Torch::NN::Module] Module to parallelize
        # @param device_ids [Array<Integer>, nil] CUDA devices (default: all available)
        # @param output_device [Integer, nil] Device for output (default: device_ids[0])
        # @param dim [Integer] Dimension to scatter inputs along (default: 0)
        def initialize(mod, device_ids: nil, output_device: nil, dim: 0)
          super()
          @module = mod
          @device_ids = device_ids || (0...Torch::CUDA.device_count).to_a
          @output_device = output_device || @device_ids.first
          @dim = dim
          @replica_losses = nil

          if @device_ids.empty?
            raise ArgumentError, "device_ids cannot be empty"
          end

          # Convert to device strings for internal use
          @device_strings = @device_ids.map { |id| "cuda:#{id}" }
          @output_device_string = "cuda:#{@output_device}"
        end

        def forward(*inputs, **kwargs)
          # Empty input check
          if inputs.empty?
            return @module.call(**kwargs)
          end

          # Single GPU fast path
          if @device_ids.size == 1
            return @module.call(*inputs.map { |i| i.to(@device_strings.first) }, **kwargs)
          end

          # Scatter inputs across devices
          scattered_inputs = scatter(inputs, @device_strings, @dim)
          scattered_kwargs = scatter_kwargs(kwargs, @device_strings, @dim)

          # Get or create replicas, sync weights
          num_replicas = scattered_inputs.size
          devices = @device_strings[0...num_replicas]
          @replicas = get_replicas(devices)
          sync_replica_weights

          # Apply in parallel
          outputs = parallel_apply(@replicas, scattered_inputs, scattered_kwargs)

          # Ensure all CUDA operations complete before gathering
          Torch::CUDA.synchronize if Torch::CUDA.available?

          # Gather outputs back to output device
          gather(outputs, @output_device_string, @dim)
        end

        # Performs backward pass on all replica losses and reduces gradients.
        # This is needed because gather creates a new tensor that breaks the
        # autograd connection across devices. By calling backward on each
        # replica's loss separately, gradients flow properly.
        #
        # @param scale [Float] Scale factor for gradients (e.g., 1.0/gradient_accumulation_steps)
        def backward(scale: 1.0)
          if @replica_losses && @replica_losses.size > 1
            # Each replica's loss contributes equally to total loss
            # Scale by 1/N to average gradients across replicas
            replica_scale = scale / @replica_losses.size

            @replica_losses.each do |replica_loss|
              # Scale the loss before backward to get properly scaled gradients
              scaled_loss = replica_loss * replica_scale
              scaled_loss.backward
            end

            @replica_losses = nil
          end

          # Reduce gradients from all replicas to the original module
          reduce_gradients
        end

        # Reduce gradients from replicas back to the original module.
        # Called automatically by backward(), but can be called manually if needed.
        def reduce_gradients
          return if @replicas.nil? || @replicas.size <= 1

          # Get original model's parameters (first replica is the original)
          original_params = @module.parameters.to_a

          # Accumulate gradients from other replicas
          @replicas[1..].each do |replica|
            replica_params = replica.parameters.to_a
            original_params.zip(replica_params).each do |orig, repl|
              next unless repl.grad

              # Move replica gradient to original device and add
              if orig.grad
                orig.grad.add!(repl.grad.to(orig.device))
              else
                orig.grad = repl.grad.to(orig.device).clone
              end
            end
          end
        end

        # Delegate training mode to wrapped module
        def train(mode = true)
          super
          @module.train(mode)
          self
        end

        def eval
          train(false)
        end

        # Delegate parameter access to wrapped module
        def parameters
          @module.parameters
        end

        def named_parameters(prefix: "", recurse: true)
          @module.named_parameters(prefix: prefix, recurse: recurse)
        end

        def state_dict(destination: nil, prefix: "")
          @module.state_dict(destination: destination, prefix: prefix)
        end

        def load_state_dict(state_dict, strict: true)
          @module.load_state_dict(state_dict, strict: strict)
        end

        def extra_inspect
          format("device_ids: %s, output_device: %s, dim: %d", @device_ids, @output_device, @dim)
        end

        private

        # Scatter a single value across devices
        def scatter_value(value, devices, dim)
          if value.is_a?(Torch::Tensor)
            NN._scatter(value, devices, dim).map(&:contiguous)
          else
            devices.map { value }
          end
        end

        def scatter(inputs, devices, dim)
          # Scatter each input, then transpose to group by device
          scattered = inputs.map { |input| scatter_value(input, devices, dim) }
          scattered.first.size.times.map { |i| scattered.map { |s| s[i] } }
        end

        def scatter_kwargs(kwargs, devices, dim)
          return devices.map { {} } if kwargs.empty?
          scattered = kwargs.transform_values { |v| scatter_value(v, devices, dim) }
          devices.size.times.map { |i| scattered.transform_values { |v| v[i] } }
        end

        def get_replicas(devices)
          # Return cached replicas if they match the requested devices
          if @replicas && @replica_devices == devices
            return @replicas
          end

          # Create new replicas and cache them
          @replica_devices = devices
          Parallel.replicate(@module, devices)
        end

        def sync_replica_weights
          return if @replicas.nil? || @replicas.size <= 1

          # Sync parameters in-place to avoid memory allocation
          original_params = @module.parameters.to_a
          @replicas[1..].each do |replica|
            replica.parameters.to_a.each_with_index do |param, i|
              param.data.copy!(original_params[i].data)
            end
          end
        end

        def parallel_apply(replicas, inputs, kwargs_list)
          Parallel.parallel_apply(replicas, inputs, kwargs_list)
        end

        def gather(outputs, target_device, dim)
          # Handle different output types
          first = outputs.first

          case first
          when Torch::Tensor
            gather_tensor(outputs, target_device, dim)
          when Array
            # Tuple/array of tensors - gather each element
            first.size.times.map do |i|
              tensors = outputs.map { |o| o[i] }
              if tensors.first.is_a?(Torch::Tensor)
                gather_tensor(tensors, target_device, dim)
              elsif tensors.first.nil?
                nil
              else
                tensors.first # Non-tensor, just return first
              end
            end
          when Hash
            # Dict of tensors - gather each value
            first.keys.map do |key|
              tensors = outputs.map { |o| o[key] }
              if tensors.first.is_a?(Torch::Tensor)
                [key, gather_tensor(tensors, target_device, dim)]
              else
                [key, tensors.first]
              end
            end.to_h
          else
            # Scalar or other - return first output
            first
          end
        end

        def gather_tensor(tensors, target_device, dim)
          first = tensors.first
          # Handle scalar tensors (0-dim) - store for backward, return average for display
          if first.dim == 0
            # Store individual losses for backward() method
            @replica_losses = tensors

            # Return mean of losses (moved to same device) for logging/display
            # Note: This returned tensor should NOT be used for backward() - use backward() instead
            # target_device is already a string like "cuda:0"
            sum = tensors.reduce(Torch.tensor(0.0, device: target_device)) do |acc, t|
              acc + t.to(target_device).detach
            end
            sum / tensors.size
          else
            NN._gather(tensors, target_device, dim)
          end
        end
      end
    end

    # Convenience alias at NN level
    DataParallel = Parallel::DataParallel
  end
end
