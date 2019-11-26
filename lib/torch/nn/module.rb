module Torch
  module NN
    class Module
      def inspect
        str = String.new
        str << "#{self.class.name}(\n"
        modules.each do |name, mod|
          str << "  (#{name}): #{mod.inspect}\n"
        end
        str << ")"
      end

      def call(*input)
        forward(*input)
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
            raise Error, "Not supported yet"
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
        modules
      end
    end
  end
end
