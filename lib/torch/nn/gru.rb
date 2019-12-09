module Torch
  module NN
    class GRU < RNNBase
      def initialize(*args, **options)
        super("GRU", *args, **options)
      end

      def run_impl(input, hx, batch_sizes)
        if batch_sizes.nil?
          Torch.gru(input, hx, _get_flat_weights, @bias, @num_layers,
                             @dropout, @training, @bidirectional, @batch_first)
        else
          Torch.gru(input, batch_sizes, hx, _get_flat_weights, @bias,
                             @num_layers, @dropout, @training, @bidirectional)
        end
      end

      def forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)
        if hx.nil?
          num_directions = @bidirectional ? 2 : 1
          hx = Torch.zeros(@num_layers * num_directions, max_batch_size, @hidden_size, dtype: input.dtype, device: input.device)
        else
          # Each batch of the hidden state should match the input sequence that
          # the user believes he/she is passing in.
          hx = permute_hidden(hx, sorted_indices)
        end

        check_forward_args(input, hx, batch_sizes)
        result = run_impl(input, hx, batch_sizes)
        output = result[0]
        hidden = result[1]
        [output, hidden]
      end

      def forward_tensor(input, hx: nil)
        batch_sizes = nil
        max_batch_size = @batch_first ? input.size(0) : input.size(1)
        sorted_indices = nil
        unsorted_indices = nil
        output, hidden = forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)
        [output, permute_hidden(hidden, unsorted_indices)]
      end

      def forward(input, hx: nil)
        forward_tensor(input, hx: hx)
      end
    end
  end
end
