module Torch
  module NN
    module Parallel
      class DistributedDataParallel < Module
        attr_reader :module, :process_group

        def initialize(mod, device_ids: nil, process_group: nil, broadcast_buffers: true)
          super()
          raise Torch::Error, "torch.distributed is not available" unless Torch::Distributed.available?

          @module = mod
          @broadcast_buffers = broadcast_buffers
          @process_group = process_group || Torch::Distributed.default_process_group
          raise Torch::Error, "Process group must be initialized before using DistributedDataParallel" unless @process_group

          @world_size = Torch::Distributed.get_world_size(@process_group)
          @rank = Torch::Distributed.get_rank(@process_group)
          @device = Array(device_ids).compact.first
          move_to_device(@device) if @device

          synchronize_parameters
          @hook_handles = register_parameter_hooks
        end

        def forward(*inputs, **kwargs)
          outputs = @module.call(*move_inputs(inputs), **move_kwargs(kwargs))
          broadcast_buffers_if_needed
          outputs
        end

        alias_method :call, :forward

        def train(mode = true)
          @module.train(mode)
          broadcast_buffers_if_needed
          self
        end

        private

        def move_to_device(device)
          return unless device

          @module.to(device)
        end

        def move_inputs(inputs)
          return inputs unless @device

          inputs.map { |value| move_value(value, @device) }
        end

        def move_kwargs(kwargs)
          return kwargs unless @device

          kwargs.transform_values { |value| move_value(value, @device) }
        end

        def move_value(value, device)
          case value
          when Torch::Tensor
            value.to(device)
          when Array
            value.map { |v| move_value(v, device) }
          when Hash
            value.transform_values { |v| move_value(v, device) }
          else
            value
          end
        end

        def synchronize_parameters
          Torch::Distributed.barrier(group: @process_group)
          @module.parameters.each do |param|
            Torch::Distributed.broadcast(param, src: 0, group: @process_group)
          end
          broadcast_buffers_if_needed
        end

        def broadcast_buffers_if_needed
          return unless @broadcast_buffers

          @module.buffers.each do |buffer|
            Torch::Distributed.broadcast(buffer, src: 0, group: @process_group)
          end
        end

        def register_parameter_hooks
          @module.parameters.filter_map do |param|
            next unless param.requires_grad?

            param.register_hook do |grad|
              Torch::Distributed.all_reduce(grad, group: @process_group)
              grad.div!(@world_size.to_f)
            end
          end
        end
      end
    end
  end
end
