module Torch
  module NN
    module Parallel
      class << self
        # Applies modules to inputs in parallel across devices.
        #
        # @param modules [Array<Torch::NN::Module>] List of module replicas
        # @param inputs [Array] List of inputs, one per module
        # @param kwargs_list [Array<Hash>, nil] Optional list of kwargs, one per module
        # @return [Array] List of outputs, one per module
        def parallel_apply(modules, inputs, kwargs_list = nil)
          kwargs_list ||= modules.map { {} }

          unless modules.size == inputs.size && modules.size == kwargs_list.size
            raise ArgumentError, "modules, inputs, and kwargs_list must have the same length"
          end

          # Single module - no parallelism needed
          if modules.size == 1
            return [apply_module(modules[0], inputs[0], kwargs_list[0])]
          end

          parallel_apply_threads(modules, inputs, kwargs_list)
        end

        private

        def parallel_apply_threads(modules, inputs, kwargs_list)
          results = Array.new(modules.size)
          errors = Array.new(modules.size)

          threads = modules.each_with_index.map do |mod, i|
            Thread.new(mod, inputs[i], kwargs_list[i], i) do |m, inp, kw, idx|
              results[idx] = apply_module(m, inp, kw)
            rescue => e
              errors[idx] = e
            end
          end

          threads.each(&:join)

          # Re-raise first error if any
          errors.each_with_index do |err, i|
            raise err if err
          end

          results
        end

        def apply_module(mod, input, kwargs)
          if input.is_a?(Array)
            mod.call(*input, **kwargs)
          else
            mod.call(input, **kwargs)
          end
        end
      end
    end
  end
end
