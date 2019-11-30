module Torch
  module NN
    class Module
      def initialize
        @training = true
        @parameters = {}
        @buffers = {}
        @modules = {}
      end

      def register_buffer(name, tensor)
        # TODO add checks
        @buffers[name] = tensor
      end

      def register_parameter(name, param)
        # TODO add checks
        @parameters[name] = param
      end

      def add_module(name, mod)
        # TODO add checks
        @modules[name] = mod
      end

      def _apply(fn)
        children.each do |mod|
          mod._apply(fn)
        end
        # TODO apply to more objects
        self
      end

      def apply(fn)
        children.each do |mod|
          mod.apply(fn)
        end
        fn.call(self)
        self
      end

      def cuda(device: nil)
        _apply ->(t) { t.cuda(device) }
      end

      def cpu
        _apply ->(t) { t.cpu }
      end

      def children
        @modules.values
      end

      def inspect
        name = self.class.name.split("::").last
        if modules.empty?
          "#{name}(#{extra_inspect})"
        else
          str = String.new
          str << "#{name}(\n"
          modules.each do |name, mod|
            str << "  (#{name}): #{mod.inspect}\n"
          end
          str << ")"
        end
      end

      def train(mode = true)
        @training = mode
        children.each do |mod|
          mod.train(mode)
        end
        self
      end

      def eval
        train(false)
      end

      def call(*input)
        forward(*input)
      end

      # modifies in-place
      def to(device)
        instance_variables.each do |name|
          param = instance_variable_get(name)
          if param.is_a?(Parameter)
            instance_variable_set(name, Parameter.new(param.to(device)))
          end
        end
        modules.each do |_, mod|
          mod.to(device)
        end
        self
      end

      def parameters
        params = []
        instance_variables.each do |name|
          param = instance_variable_get(name)
          params << param if param.is_a?(Parameter)
        end
        params + modules.flat_map { |_, mod| mod.parameters }
      end

      def zero_grad
        parameters.each do |param|
          if param.grad
            param.grad.detach!
            param.grad.zero!
          end
        end
      end

      def method_missing(method, *args, &block)
        modules[method.to_s] || super
      end

      def respond_to?(method, include_private = false)
        modules.key?(method.to_s) || super
      end

      private

      def modules
        modules = {}
        instance_variables.each do |name|
          mod = instance_variable_get(name)
          modules[name[1..-1]] = mod if mod.is_a?(Module)
        end
        @modules.merge(modules)
      end

      def extra_inspect
        nil
      end

      def format(str, vars)
        str % vars.map(&:inspect)
      end
    end
  end
end
