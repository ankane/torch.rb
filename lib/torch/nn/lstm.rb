module Torch
  module NN
    class LSTM < RNNBase
      def initialize(*args, **options)
        super("LSTM", *args, **options)
      end

      def check_forward_args(input, hidden, batch_sizes)
        check_input(input, batch_sizes)
        expected_hidden_size = get_expected_hidden_size(input, batch_sizes)

        # TODO pass message
        check_hidden_size(hidden[0], expected_hidden_size)
        check_hidden_size(hidden[1], expected_hidden_size)
      end

      def permute_hidden(hx, permutation)
        if permutation.nil?
          return hx
        end
        raise NotImplementedYet
      end

      def forward_impl(input, hx, batch_sizes, max_batch_size, sorted_indices)
        if hx.nil?
          num_directions = @bidirectional ? 2 : 1
          zeros = Torch.zeros(@num_layers * num_directions, max_batch_size, @hidden_size, dtype: input.dtype, device: input.device)
          hx = [zeros, zeros]
        else
          # Each batch of the hidden state should match the input sequence that
          # the user believes he/she is passing in.
          hx = permute_hidden(hx, sorted_indices)
        end

        check_forward_args(input, hx, batch_sizes)
        if batch_sizes.nil?
          result = Torch.lstm(input, hx, _get_flat_weights, @bias, @num_layers,
                              @dropout, @training, @bidirectional, @batch_first)
        else
          result = Torch.lstm(input, batch_sizes, hx, _get_flat_weights, @bias,
                              @num_layers, @dropout, @training, @bidirectional)
        end
        output = result[0]
        hidden = result[1..-1]

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
        # TODO PackedSequence
        forward_tensor(input, hx: hx)
      end
    end
  end
end
