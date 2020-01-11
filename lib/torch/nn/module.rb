module Torch
  module NN
    class Module
      include Utils

      def initialize
        @training = true
        @parameters = {}
        @buffers = {}
        @modules = {}
      end

      def forward
        raise NotImplementedError
      end

      def register_buffer(name, tensor)
        # TODO add checks
        @buffers[name] = tensor
        instance_variable_set("@#{name}", tensor)
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

      def type(dst_type)
        _apply ->(t) { t.type(dst_type) }
      end

      def float
        _apply ->(t) { t.floating_point? ? t.float : t }
      end

      def double
        _apply ->(t) { t.floating_point? ? t.double : t }
      end

      def half
        _apply ->(t) { t.floating_point? ? t.half : t }
      end

      # modifies in-place
      def to(device)
        convert = lambda do |t|
          t.to(device)
        end

        _apply(convert)
      end

      def call(*input, **kwargs)
        forward(*input, **kwargs)
      end

      def state_dict(destination: nil)
        destination ||= {}
        named_parameters.each do |k, v|
          destination[k] = v
        end
        destination
      end

      def load_state_dict(state_dict)
        raise NotImplementedYet
      end

      def parameters
        named_parameters.values
      end

      def named_parameters(prefix: "", recurse: true)
        params = {}
        if recurse
          named_children.each do |name, mod|
            params.merge!(mod.named_parameters(prefix: "#{name}.", recurse: recurse))
          end
        end
        instance_variables.each do |name|
          param = instance_variable_get(name)
          params[[prefix, name[1..-1]].join] = param if param.is_a?(Parameter)
        end
        @parameters.each do |name, param|
          params[[prefix, name].join] = param
        end
        params
      end

      def buffers
        named_buffers.values
      end

      def named_buffers
        @buffers || {}
      end

      def children
        named_children.values
      end

      def named_children
        modules = {}
        instance_variables.each do |name|
          mod = instance_variable_get(name)
          modules[name[1..-1]] = mod if mod.is_a?(Module)
        end
        @modules.each do |name, mod|
          modules[name] = mod
        end
        modules
      end

      def modules
        named_modules.values
      end

      def named_modules
        {"" => self}.merge(named_children)
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

      def requires_grad!(requires_grad: true)
        parameters.each do |p|
          p.requires_grad!(requires_grad)
        end
        self
      end

      def zero_grad
        parameters.each do |param|
          if param.grad
            param.grad.detach!
            param.grad.zero!
          end
        end
      end

      def share_memory
        _apply ->(t) { t.share_memory! }
      end

      def inspect
        name = self.class.name.split("::").last
        if children.empty?
          "#{name}(#{extra_inspect})"
        else
          str = String.new
          str << "#{name}(\n"
          children.each do |name, mod|
            str << "  (#{name}): #{mod.inspect}\n"
          end
          str << ")"
        end
      end

      def method_missing(method, *args, &block)
        name = method.to_s
        if named_parameters.key?(name)
          named_parameters[name]
        elsif named_buffers.key?(name)
          named_buffers[name]
        elsif named_modules.key?(name)
          named_modules[name]
        else
          super
        end
      end

      def respond_to?(method, include_private = false)
        name = method.to_s
        named_parameters.key?(name) || named_buffers.key?(name) || named_modules.key?(name) || super
      end

      private

      def extra_inspect
        nil
      end

      def format(str, *vars, **options)
        vars =
          if vars.any?
            vars.map(&:inspect)
          else
            options.map { |k, v| [k, v.inspect] }.to_h
          end
        str % vars
      end

      def dict
        instance_variables.map { |k| [k[1..-1].to_sym, instance_variable_get(k)] }.to_h
      end
    end
  end
end
