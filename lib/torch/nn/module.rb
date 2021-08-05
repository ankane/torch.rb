module Torch
  module NN
    class Module
      include Utils

      attr_reader :training

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

        instance_variables.each do |key|
          param = instance_variable_get(key)
          if param.is_a?(Parameter)
            param_applied = nil
            Torch.no_grad do
              param_applied = fn.call(param)
            end
            # TODO should_use_set_data
            instance_variable_set(key, Parameter.new(param_applied, requires_grad: param.requires_grad))

            if param.grad
              grad_applied = nil
              Torch.no_grad do
                grad_applied = fn.call(param.grad)
              end
              # TODO should_use_set_data
              instance_variable_get(key).grad = grad_applied.requires_grad!(param.grad.requires_grad)
            end
          end
        end

        @buffers.each_key do |k|
          buf = @buffers[k]
          unless buf.nil?
            @buffers[k] = fn.call(buf)
            instance_variable_set("@#{k}", @buffers[k])
          end
        end

        self
      end

      def apply(fn)
        children.each do |mod|
          mod.apply(fn)
        end
        fn.call(self)
        self
      end

      # TODO add device
      def cuda
        _apply ->(t) { t.cuda }
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

      def state_dict(destination: nil, prefix: "")
        destination ||= {}
        save_to_state_dict(destination, prefix: prefix)

        named_children.each do |name, mod|
          next unless mod
          mod.state_dict(destination: destination, prefix: prefix + name + ".")
        end
        destination
      end

      def load_state_dict(state_dict, strict: true)
        # TODO support strict: false
        raise "strict: false not implemented yet" unless strict

        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # TODO handle metadata

        _load = lambda do |mod, prefix = ""|
          # TODO handle metadata
          local_metadata = {}
          mod.send(:load_from_state_dict, state_dict, prefix, local_metadata, true, missing_keys, unexpected_keys, error_msgs)
          mod.named_children.each do |name, child|
            _load.call(child, prefix + name + ".") unless child.nil?
          end
        end

        _load.call(self)

        if strict
          if unexpected_keys.any?
            error_msgs << "Unexpected key(s) in state_dict: #{unexpected_keys.join(", ")}"
          end

          if missing_keys.any?
            error_msgs << "Missing key(s) in state_dict: #{missing_keys.join(", ")}"
          end
        end

        if error_msgs.any?
          # just show first error
          raise Error, error_msgs[0]
        end

        nil
      end

      def parameters
        named_parameters.values
      end

      def named_parameters(prefix: "", recurse: true)
        params = {}
        if recurse
          named_children.each do |name, mod|
            params.merge!(mod.named_parameters(prefix: "#{prefix}#{name}.", recurse: recurse))
          end
        end
        instance_variables.each do |name|
          param = instance_variable_get(name)
          params[[prefix, name[1..-1]].join] = param if param.is_a?(Parameter)
        end
        @parameters.each do |name, param|
          params[[prefix, name].join] = param if param
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

      # TODO return enumerator?
      def named_modules(memo: nil, prefix: "")
        ret = {}
        memo ||= Set.new
        unless memo.include?(self)
          memo << self
          ret[prefix] = self
          named_children.each do |name, mod|
            next unless mod.is_a?(Module)
            submodule_prefix = prefix + (!prefix.empty? ? "." : "") + name
            mod.named_modules(memo: memo, prefix: submodule_prefix).each do |m|
              ret[m[0]] = m[1]
            end
          end
        end
        ret
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
        if named_children.empty?
          "#{name}(#{extra_inspect})"
        else
          str = String.new
          str << "#{name}(\n"
          named_children.each do |name, mod|
            mod_str = mod.inspect
            mod_str = mod_str.lines.join("  ")
            str << "  (#{name}): #{mod_str}\n"
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
        elsif method.end_with?("=") && named_modules.key?(method[0..-2])
          if instance_variable_defined?("@#{method[0..-2]}")
            instance_variable_set("@#{method[0..-2]}", *args)
          else
            raise NotImplementedYet
          end
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

      # used for format
      # remove tensors for performance
      # so we can skip call to inspect
      def dict
        instance_variables.reject { |k| instance_variable_get(k).is_a?(Tensor) }.map { |k| [k[1..-1].to_sym, instance_variable_get(k)] }.to_h
      end

      def load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        # TODO add hooks

        # TODO handle non-persistent buffers
        persistent_buffers = named_buffers
        local_name_params = named_parameters(recurse: false).merge(persistent_buffers)
        local_state = local_name_params.select { |_, v| !v.nil? }

        local_state.each do |name, param|
          key = prefix + name
          if state_dict.key?(key)
            input_param = state_dict[key]

            # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
            if param.shape.length == 0 && input_param.shape.length == 1
              input_param = input_param[0]
            end

            if input_param.shape != param.shape
              # local shape should match the one in checkpoint
              error_msgs << "size mismatch for #{key}: copying a param with shape #{input_param.shape} from checkpoint, " +
                            "the shape in current model is #{param.shape}."
              next
            end

            begin
              Torch.no_grad do
                param.copy!(input_param)
              end
            rescue => e
              error_msgs << "While copying the parameter named #{key.inspect}, " +
                            "whose dimensions in the model are #{param.size} and " +
                            "whose dimensions in the checkpoint are #{input_param.size}, " +
                            "an exception occurred: #{e.inspect}"
            end
          elsif strict
            missing_keys << key
          end
        end

        if strict
          state_dict.each_key do |key|
            if key.start_with?(prefix)
              input_name = key[prefix.length..-1]
              input_name = input_name.split(".", 2)[0]
              if !named_children.key?(input_name) && !local_state.key?(input_name)
                unexpected_keys << key
              end
            end
          end
        end
      end

      def save_to_state_dict(destination, prefix: "")
        named_parameters(recurse: false).each do |k, v|
          destination[prefix + k] = v
        end
        named_buffers.each do |k, v|
          destination[prefix + k] = v
        end
      end
    end
  end
end
