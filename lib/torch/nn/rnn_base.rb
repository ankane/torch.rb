module Torch
  module NN
    class RNNBase < Module
      def initialize(
        mode,
        input_size,
        hidden_size,
        num_layers: 1,
        bias: true,
        batch_first: false,
        dropout: 0.0,
        bidirectional: false
      )
        super()
        @mode = mode
        @input_size = input_size
        @hidden_size = hidden_size
        @num_layers = num_layers
        @bias = bias
        @batch_first = batch_first
        @dropout = dropout.to_f
        @bidirectional = bidirectional
        num_directions = bidirectional ? 2 : 1

        if !dropout.is_a?(Numeric) || !(dropout >= 0 && dropout <= 1)
          raise ArgumentError, "dropout should be a number in range [0, 1] " +
                               "representing the probability of an element being " +
                               "zeroed"
        end
        if dropout > 0 && num_layers == 1
          warn "dropout option adds dropout after all but last " +
               "recurrent layer, so non-zero dropout expects " +
               "num_layers greater than 1, but got dropout=#{dropout} and " +
               "num_layers=#{num_layers}"
        end

        gate_size =
          case mode
          when "LSTM"
            4 * hidden_size
          when "GRU"
            3 * hidden_size
          when "RNN_TANH"
            hidden_size
          when "RNN_RELU"
            hidden_size
          else
            raise ArgumentError, "Unrecognized RNN mode: #{mode}"
          end

        @all_weights = []
        num_layers.times do |layer|
          num_directions.times do |direction|
            layer_input_size = layer == 0 ? input_size : hidden_size * num_directions

            w_ih = Parameter.new(Torch::Tensor.new(gate_size, layer_input_size))
            w_hh = Parameter.new(Torch::Tensor.new(gate_size, hidden_size))
            b_ih = Parameter.new(Torch::Tensor.new(gate_size))
            # Second bias vector included for CuDNN compatibility. Only one
            # bias vector is needed in standard definition.
            b_hh = Parameter.new(Torch::Tensor.new(gate_size))
            layer_params = [w_ih, w_hh, b_ih, b_hh]

            suffix = direction == 1 ? "_reverse" : ""
            param_names = ["weight_ih_l%s%s", "weight_hh_l%s%s"]
            if bias
              param_names += ["bias_ih_l%s%s", "bias_hh_l%s%s"]
            end
            param_names.map! { |x| x % [layer, suffix] }

            param_names.zip(layer_params) do |name, param|
              instance_variable_set("@#{name}", param)
            end
            @all_weights << param_names
          end
        end

        flatten_parameters
        reset_parameters
      end

      def flatten_parameters
        # no-op unless module is on the GPU and cuDNN is enabled
      end

      def _apply(fn)
        ret = super
        flatten_parameters
        ret
      end

      def reset_parameters
        stdv = 1.0 / Math.sqrt(@hidden_size)
        parameters.each do |weight|
          Init.uniform!(weight, a: -stdv, b: stdv)
        end
      end

      def permute_hidden(hx, permutation)
        if permutation.nil?
          return hx
        end
        raise NotImplementedYet
      end

      def forward(input, hx: nil)
        is_packed = false # TODO isinstance(input, PackedSequence)
        if is_packed
          input, batch_sizes, sorted_indices, unsorted_indices = input
          max_batch_size = batch_sizes[0]
          max_batch_size = max_batch_size.to_i
        else
          batch_sizes = nil
          max_batch_size = @batch_first ? input.size(0) : input.size(1)
          sorted_indices = nil
          unsorted_indices = nil
        end

        if hx.nil?
          num_directions = @bidirectional ? 2 : 1
          hx = Torch.zeros(@num_layers * num_directions, max_batch_size,
            @hidden_size, dtype: input.dtype, device: input.device)
        else
          # Each batch of the hidden state should match the input sequence that
          # the user believes he/she is passing in.
          hx = permute_hidden(hx, sorted_indices)
        end

        check_forward_args(input, hx, batch_sizes)
        _rnn_impls = {
          "RNN_TANH" => Torch.method(:rnn_tanh),
          "RNN_RELU" => Torch.method(:rnn_relu)
        }
        _impl = _rnn_impls[@mode]
        if batch_sizes.nil?
          result = _impl.call(input, hx, _get_flat_weights, @bias, @num_layers,
                           @dropout, @training, @bidirectional, @batch_first)
        else
          result = _impl.call(input, batch_sizes, hx, _get_flat_weights, @bias,
                           @num_layers, @dropout, @training, @bidirectional)
        end
        output = result[0]
        hidden = result[1]

        if is_packed
          raise NotImplementedYet
          # output = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
        end
        [output, permute_hidden(hidden, unsorted_indices)]
      end

      # TODO add more parameters
      def extra_inspect
        s = String.new("%{input_size}, %{hidden_size}")
        if @num_layers != 1
          s += ", num_layers: %{num_layers}"
        end
        format(s, input_size: @input_size, hidden_size: @hidden_size, num_layers: @num_layers)
      end

      private

      def _flat_weights
        @all_weights.flatten.map { |v| instance_variable_get("@#{v}") }.compact
      end

      def _get_flat_weights
        _flat_weights
      end

      def check_input(input, batch_sizes)
        expected_input_dim = !batch_sizes.nil? ? 2 : 3
        if input.dim != expected_input_dim
          raise ArgumentError, "input must have #{expected_input_dim} dimensions, got #{input.dim}"
        end
        if @input_size != input.size(-1)
          raise ArgumentError, "input.size(-1) must be equal to input_size. Expected #{@input_size}, got #{input.size(-1)}"
        end
      end

      def get_expected_hidden_size(input, batch_sizes)
        if !batch_sizes.nil?
          mini_batch = batch_sizes[0]
          mini_batch = mini_batch.to_i
        else
          mini_batch = @batch_first ? input.size(0) : input.size(1)
        end
        num_directions = @bidirectional ? 2 : 1
        [@num_layers * num_directions, mini_batch, @hidden_size]
      end

      def check_hidden_size(hx, expected_hidden_size)
        if hx.size != expected_hidden_size
          raise ArgumentError, "Expected hidden size #{expected_hidden_size.inspect}, got #{hx.size.inspect}"
        end
      end

      def check_forward_args(input, hidden, batch_sizes)
        check_input(input, batch_sizes)
        expected_hidden_size = get_expected_hidden_size(input, batch_sizes)
        check_hidden_size(hidden, expected_hidden_size)
      end
    end
  end
end
